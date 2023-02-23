#include <map>
#include <vector>
#include <string>
#include <math.h>
#include <fstream>
#include <sstream>

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_opengl.h>

#include <GLES2/gl2.h>

#define CHUNK_WIDTH 16
#define CHUNK_HEIGHT 16
#define CHUNK_DEPTH 16

#define COLLECTION_WIDTH 16
#define COLLECTION_HEIGHT 2
#define COLLECTION_DEPTH 16

// Equivalent to cw * ch * cd
#define COMBINED_SIZE (CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_DEPTH)
#define VBO_SIZE (COLLECTION_WIDTH * COLLECTION_HEIGHT * COLLECTION_DEPTH)

const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 480;
SDL_Window *windows;

static const int transparency[16] = { 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 4, 0 };
// Notice how a block's type might be different from the vertices it has
enum class BlockTypes : uint8_t {
    air = 0,
    grass = 1,
    woodLog = 2,
    leaves = 3,
    dirt = 4,
    stone = 5,
    cobblestone = 6,
    woodenPlanks = 7,
    window = 8,
    glass = 9,
    bricks = 10,
    stoneBricks = 11,
    water = 12,
    sand = 13,
    
    TOTAL = 14
};

// A very naive way of getting the (texture) index of a block type
uint8_t get_index_from_type(BlockTypes type) {
    switch (type) {
        case BlockTypes::air: return 0;
        case BlockTypes::grass: return 1;
        case BlockTypes::woodLog: return 2;
        case BlockTypes::leaves: return 3;
        case BlockTypes::dirt: return 4;
        case BlockTypes::stone: return 6;
        case BlockTypes::cobblestone: return 7;
        case BlockTypes::woodenPlanks: return 9;
        case BlockTypes::window: return 10;
        case BlockTypes::glass: return 11;
        case BlockTypes::bricks: return 12;
        case BlockTypes::stoneBricks: return 13;
        case BlockTypes::water: return 14;
        case BlockTypes::sand: return 15;
        
        default: return 0;
    }
};

// Modified Cxxdroid's load texture function
static SDL_Surface *load_surface(const char *path)
{
    SDL_Surface *img = IMG_Load(path);
    if (img == NULL)
    {
        fprintf(stderr, "IMG_Load Error: %s\n", IMG_GetError());
        return NULL;
    }
    return img;
}


struct Vec3f {
    float x, y, z;
    void set_zero() {
        x = 0;
        y = 0;
        z = 0;
    }
    float dot_prod(Vec3f &other) {
        return x * other.x + y * other.y + z * other.z;
    }
    Vec3f cross_prod(Vec3f &other) { 
        float cx = y * other.z - z * other.y;
        float cy = z * other.x - x * other.z;
        float cz = x * other.y - y * other.x;
        
        Vec3f result = { cx, cy, cz };
        
        return result;
    }  
    float len() {
        return sqrt(x*x + y*y + z*z);
    }
    void multiply(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
    }
    void norm() {
        multiply(1 / len());
    }
};

struct Vec4f {
    float x, y, z, w;
    float dst(Vec4f &other) {
        float dx = other.x - x;
        float dy = other.y - y;
        float dz = other.z - z;
        float dw = other.w - w;
        
        return sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
    }
};

class Mat4x4 {
    public:
       int M00 = 0,  M10 = 1,  M20 = 2,  M30 = 3,
           M01 = 4,  M11 = 5,  M21 = 6,  M31 = 7,
           M02 = 8,  M12 = 9,  M22 = 10, M32 = 11,
           M03 = 12, M13 = 13, M23 = 14, M33 = 15;
           
       float values[4 * 4] = { 0 };
       
       Mat4x4() {
           this->identity();
       }
       void identity() {
           for (int i = 0; i < 4 * 4; i++) {
               values[i] = 0.0f;
           }
           values[M00] = 1.0f;
           values[M11] = 1.0f;
           values[M22] = 1.0f;
           values[M33] = 1.0f;
       }
       void set_perspective(float fovDegrees, float zNear, float zFar, float aspectRatio) {            
           float fovR = float(1.0f / tan(fovDegrees * (M_PI / 180.0f) / 2.0f));           
           float range = zFar - zNear;
           
           identity();                    
           values[M00] = fovR / aspectRatio;            
           values[M11] = fovR;            
           
           values[M22] = -(zFar + zNear) / range;            
           values[M32] = -(2 * zFar * zNear) / range;           
           values[M23] = -1.0f;            
           values[M33] = 0.0f;
       }
       void set_look_at(Vec3f cameraPosition, Vec3f lookingAt, Vec3f up) {
           Vec3f fwd = { cameraPosition.x - lookingAt.x,
                          cameraPosition.y - lookingAt.y,
                          cameraPosition.z - lookingAt.z };
           fwd.norm();
           
           Vec3f cameraXAxis = fwd.cross_prod(up);
           cameraXAxis.norm();
           
           Vec3f cameraYAxis = cameraXAxis.cross_prod(fwd);
           
           identity();
           
           values[M00] = cameraXAxis.x;
           values[M10] = cameraXAxis.y;
           values[M20] = cameraXAxis.z;
     
           values[M01] = cameraYAxis.x;
           values[M11] = cameraYAxis.y;
           values[M21] = cameraYAxis.z;
           
           values[M02] = fwd.x;
           values[M12] = fwd.y;
           values[M22] = fwd.z;
           
           values[M30] = -cameraXAxis.dot_prod(cameraPosition);
           values[M31] = -cameraYAxis.dot_prod(cameraPosition);
           values[M32] = -fwd.dot_prod(cameraPosition);
           
       }
       
       
       void set_translation(float x, float y, float z) {
           identity();
           
           values[M30] = x;
           values[M31] = y;
           values[M32] = z;
       }
       void set_translation(Vec3f to) {
           set_translation(to.x, to.y, to.z);
       }
       
       void set_scaling(float x, float y, float z) {
           identity();
           
           values[M00] = x;
           values[M11] = y;
           values[M22] = z;
       }
       void set_scaling(Vec3f to) {
           set_scaling(to.x, to.y, to.z);
       }
       void set_rotationY(float radians) {
           identity();
           
           values[M00] = cos(radians);
           values[M20] = sin(radians);
           values[M02] = -sin(radians);
           values[M22] = cos(radians);
       }
       // With as the right-hand matrix
       Mat4x4 multiply(Mat4x4 &with) {
           Mat4x4 r;
           
           int row = 0;
           int column = 0;
           for (int i = 0; i < 16; i++) {
               row = (i / 4) * 4;
               column = (i % 4);
               r.values[i] = (values[row + 0] * with.values[column + 0]) +
                             (values[row + 1] * with.values[column + 4]) +
                             (values[row + 2] * with.values[column + 8]) +
                             (values[row + 3] * with.values[column + 12]);
           }
           
           return r;
       }
       Vec3f multiply_vector(Vec3f &src) {
           Vec3f result;
           result.x = src.x * values[M00] + src.y * values[M10] + src.z * values[M20] + values[M30];
           result.y = src.x * values[M01] + src.y * values[M11] + src.z * values[M21] + values[M31];
           result.z = src.x * values[M02] + src.y * values[M12] + src.z * values[M22] + values[M32];
           float w = src.x * values[M03] + src.y * values[M13] + src.z * values[M23] + values[M33];
           
           if (w != 0.0f) {
               result.x /= w;
               result.y /= w;
               result.z /= w;
           }
           
           return result;
       }
       Vec4f multiply_vector(Vec4f &src) {
           Vec4f result;
           result.x = src.x * values[M00] + src.y * values[M01] + src.z * values[M02] + src.w * values[M03];
           result.y = src.x * values[M10] + src.y * values[M11] + src.z * values[M12] + src.w * values[M13];
           result.z = src.x * values[M20] + src.y * values[M21] + src.z * values[M22] + src.w * values[M23];
           result.w = src.x * values[M30] + src.y * values[M31] + src.z * values[M23] + src.w * values[M33];
           
           return result;
       }
       // Print the column-major matrix
       std::string to_string() {
           return 
           "[ " + std::to_string(values[M00]) + "|" + std::to_string(values[M10]) + "|" + std::to_string(values[M20]) + "|" + std::to_string(values[M30]) + "|\n  " +
                 std::to_string(values[M01]) + "|" + std::to_string(values[M11]) + "|" + std::to_string(values[M21]) + "|" + std::to_string(values[M31]) + "|\n  " +
                 std::to_string(values[M02]) + "|" + std::to_string(values[M12]) + "|" + std::to_string(values[M22]) + "|" + std::to_string(values[M32]) + "|\n  " + 
                 std::to_string(values[M03]) + "|" + std::to_string(values[M13]) + "|" + std::to_string(values[M23]) + "|" + std::to_string(values[M33]) + " ]\n";
       }
};


namespace Projection {
    Vec3f cameraPosition;
    Vec3f lookingAt;
    Vec3f lookDirection;
    
    float fov;
    float zNear, zFar;
    float aspectRatio;
    
    Mat4x4 viewMat;
    Mat4x4 projMat;
    Mat4x4 combined;
    
    void update() {
        Vec3f up = { 0.0f, 1.0f, 0.0f };
        viewMat.set_look_at(cameraPosition, lookingAt, up);
        combined = viewMat.multiply(projMat);
    }
    void reset() {
        cameraPosition.set_zero();
        cameraPosition = { 0.0f, 0.0f, 0.0f };
        
        fov = 90.0f;
        zNear = 0.1f;
        zFar = 1000.0f;
        aspectRatio = float(SCREEN_WIDTH) / float(SCREEN_HEIGHT);
        projMat.set_perspective(fov, zNear, zFar, aspectRatio);
    }
};



class Shader {
    public:
       const char *vertexFile, *fragmentFile;
       
       Shader(const char *vertexFile, const char *fragmentFile) {
            this->vertexFile = vertexFile;
            this->fragmentFile = fragmentFile;
            
            std::string vertContent, fragContent;
               
            // Read from the vertex shader source
            std::ifstream vert;
            std::ifstream frag;
            vert.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            frag.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            try {
                vert.open(vertexFile);
                frag.open(fragmentFile);
                
                std::stringstream stringVert, stringFrag;
                stringVert << vert.rdbuf();
                stringFrag << frag.rdbuf();
                
                vert.close();
                frag.close();
                  
                vertContent = stringVert.str();
                fragContent = stringFrag.str();
             } catch (std::ifstream::failure except) {
                printf("Couldn't open the shader file: %s\n", except.what());
             }
             
             
             this->vertex = vertContent.c_str();
             this->fragment = fragContent.c_str();
             
             
             this->load(this->vertex, this->fragment);
       }
       
       void load(const char *vertSource, const char *fragSource) {
            int check;
            char log[512];
            
            
            GLuint vert = glCreateShader(GL_VERTEX_SHADER);
        	glShaderSource(vert, 1, &vertSource, NULL);
            glCompileShader(vert);
               
            glGetShaderiv(vert, GL_COMPILE_STATUS, &check); 
            if (!check) {
                 glGetShaderInfoLog(vert, 512, NULL, log);
                 printf("%s\n", log);
            }
            
            
            GLuint fragm = glCreateShader(GL_FRAGMENT_SHADER);
        	glShaderSource(fragm, 1, &fragSource, NULL);
            glCompileShader(fragm);
             
            glGetShaderiv(fragm, GL_COMPILE_STATUS, &check); 
            if (!check) {
                glGetShaderInfoLog(fragm, 512, NULL, log);
                printf("%s\n", log);
            }
               
            this->program = glCreateProgram();
            glAttachShader(this->program, vert);
            glAttachShader(this->program, fragm);
            glLinkProgram(this->program);
               
            glGetProgramiv(this->program, GL_LINK_STATUS, &check);
            if (!check) {
                glGetProgramInfoLog(this->program, 512, NULL, log);
                printf("%s\n", log);
            }
            glDeleteShader(vert);
            glDeleteShader(fragm);
       }
       void use() {
           glUseProgram(this->program);
       }
       GLint attribute_location(const char *name) {
           return glGetAttribLocation(program, name);
       }
       GLint uniform_location(const char *name) {
           return glGetUniformLocation(program, name);
       }
       GLuint get_program() {
           return program;
       }
       
       void set_uniform_int(const char *name, int value) {
           glUniform1i(this->uniform_location(name), value);
       }
       void set_uniform_bool(const char *name, bool value) {
           glUniform1i(this->uniform_location(name), (int)value);
       }
       void set_uniform_float(const char *name, float value) {
           glUniform1f(this->uniform_location(name), value);
       }
       void set_uniform_vec2f(const char *name, float x, float y) {
           glUniform2f(this->uniform_location(name), x, y);
       }
       void set_uniform_vec3f(const char *name, float x, float y, float z) {
           glUniform3f(this->uniform_location(name), x, y, z);
       }
       void set_uniform_vec4f(const char *name, float x, float y, float z, float t) {
           glUniform4f(this->uniform_location(name), x, y, z, t);
       }
       void set_uniform_mat4(const char *name, Mat4x4 input) {
           glUniformMatrix4fv(this->uniform_location(name), 1, GL_FALSE, input.values);
       }
    protected:
       const char *vertex;
       const char *fragment;
       
       GLuint program;
};
    
class Shaders {
    std::map<const char*, Shader*> shaders;
    public:
        static Shaders& get()
        {
            static Shaders ins;
            return ins;
        }
        void clear();
        void load();
        void use(const char *name) {
            Shader *s = find(name);
            s->use();
        }
        Shader *find(const char *name) {
            return shaders[name];
        }
    private:
        Shaders() {}
        ~Shaders() {}
    public:
        Shaders(Shaders const&) = delete;
        void operator = (Shaders const&) = delete;
};
void Shaders::load() {
    auto s = [&](const char *name, const char *vertexFile, const char *fragmentFile) {
        Shader *shader = new Shader(vertexFile, fragmentFile); 
        shaders[name] = shader;
    };
       
    s("testShader", "testShader.vert", "testShader.frag");
    s("overlayShader", "overlayShader.vert", "overlayShader.frag");
    s("uiShader", "uiShader.vert", "uiShader.frag");	
};

void Shaders::clear() {
    for (auto &shader : shaders) {
        Shader *second = shader.second;
        glDeleteProgram(second->get_program());
    }
};

class Texture {
    public:
       Texture(const char *fileName, bool repeating) {
           this->fileName = fileName;
           
           this->texture = nullptr;              
           this->textureIndex = 0;
           this->loaded = false;
           this->repeating = repeating;
       };
       
       Texture(const char *fileName) : Texture(fileName, false) {};
       
       void load() {
           if (!loaded) {
               glGenTextures(1, &this->textureIndex);
               glBindTexture(GL_TEXTURE_2D, this->textureIndex);
               
               this->texture = load_surface(this->fileName);
               if (this->texture != NULL) {
                   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, this->texture->w, this->texture->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, this->texture->pixels);
                   glGenerateMipmap(GL_TEXTURE_2D);
                   
                   if (repeating) {
                       glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                       glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                   }
                   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                   
               } else {
                   printf("Counldn't generate texture!\n");
               }
               SDL_FreeSurface(this->texture);
               
               this->loaded = true;
               printf("Loaded texture.\n");
           }
       }
       void draw() { 
           glActiveTexture(GL_TEXTURE0);
           glBindTexture(GL_TEXTURE_2D, this->textureIndex);
       }
       void clear() {
           glDeleteTextures(1, &this->textureIndex);
       }
    protected:
       const char *fileName;
       
       unsigned int textureIndex;
       SDL_Surface *texture;
       
       bool loaded;
       bool repeating;
};


// A vector representing 4 bytes of memory.
// In our game, the x, y, z components get interpreted like the position of a vertice,
// while w means the type of that vertice
struct Vec4b {
    uint8_t x, y, z, w;
    Vec4b() {}
    Vec4b(uint8_t x, uint8_t y, uint8_t z, uint8_t w) : x(x), y(y), z(z), w(w) {}
};

struct Vec2i {
    int x, y;
};

struct Vec2f {
    float x, y;
    Vec2f() {}
    Vec2f(float x, float y) : x(x), y(y) {}
};

struct Vec3i {
    int x, y, z;
    Vec3i() {}
    Vec3i(int x, int y, int z) : x(x), y(y), z(z) {}
};

struct Vec3b {
    uint8_t x, y, z;
    Vec3b() {}
    Vec3b(uint8_t x, uint8_t y, uint8_t z) : x(x), y(y), z(z) {}
};

struct AABB {
    Vec3f min, max;
    
    // Translate each min and max vectors by a respective vector
    AABB add(Vec3f &position) {
        AABB result;
        
        result.min.x = min.x + position.x;
        result.min.y = min.y + position.y; 
        result.min.z = min.z + position.z; 
        
        result.max.x = max.x + position.x;
        result.max.y = max.y + position.y;
        result.max.z = max.z + position.z;
        
        return result;
    }
};

struct Block {
    // The position relative to this chunk's position in 8 bits (-256 <= value <= 256)
    Vec3b position;
    BlockTypes type;
    
    Block() {}
    Block(uint8_t x, uint8_t y, uint8_t z, BlockTypes type) : position(x, y, z), type(type) {}
    
    AABB get_AABB(Vec3b offset) {
        Vec3f m1 = { offset.x - 0.5f, offset.y - 0.5f, offset.z - 0.5f };
        Vec3f m2 = { offset.x + 0.5f, offset.y + 0.5f, offset.z + 0.5f };
        AABB result = { m1, m2 };
        
        return result;
    }
};

// Perlin noise utilities
namespace Perlin {
    uint8_t perms[256];
    float interp_1D(float value1, float value2, float alpha) {
         //return (value2 - value1) * alpha + value1;
         return (value2 - value1) * alpha + value1;
    }
    
    float fade(float val) {
         return (3.0 - val * 2.0) * val * val;
         //return 1;
    }
    
    float dot(Vec2f v1, Vec2f v2) {
         return (v1.x * v2.x + v1.y * v2.y);
    }
    
    void load_perm_table() {
         memset(perms, 0, sizeof(perms));
        
         for (int i = 0; i < 256; i++) {
              perms[i] = i;
         }
         
         // Shuffle element indices
         for (int i = 255; i > 0; i--) {
              int index = rand() % 255;
              uint8_t temporary = perms[i];
              
              perms[i] = perms[index];
              perms[index] = temporary;
         }
    }
    
    Vec2f get_vector_from_perm(uint8_t value) {
        uint8_t c = value & 3;
        if (c == 0) {
            return Vec2f(1.0f, 1.0f);
        }
        if (c == 1) {
            return Vec2f(-1.0f, 1.0f);
        }
        if (c == 2) {
            return Vec2f(-1.0f, -1.0f);
        }
        
        return Vec2f(1.0f, -1.0f);
    } 
    
    float noise(float nx, float ny) {
         int x = (int)floor(nx);
         int y = (int)floor(ny);
         int px = x + 1;
         int py = y + 1;
         
         float dx = nx - (float)x;
         float dy = ny - (float)y;
         
         Vec2f corner1 = Vec2f(dx - 1.0f, dy - 1.0f);
         Vec2f corner2 = Vec2f(dx, dy - 1.0f);
         Vec2f corner3 = Vec2f(dx - 1.0f, dy);
         Vec2f corner4 = Vec2f(dx, dy);
         
         uint8_t vCorner1 = perms[perms[px] + py];
         uint8_t vCorner2 = perms[perms[x] + py];
         uint8_t vCorner3 = perms[perms[px] + y];
         uint8_t vCorner4 = perms[perms[x] + y];
         
         float dot1 = dot(corner1, get_vector_from_perm(vCorner1));
         float dot2 = dot(corner2, get_vector_from_perm(vCorner2));
         float dot3 = dot(corner3, get_vector_from_perm(vCorner3));
         float dot4 = dot(corner4, get_vector_from_perm(vCorner4));
         
         float fx = fade(dx);
         float fy = fade(dy);
         
         float value = 0.0f;
         value = interp_1D(fx, interp_1D(fy, dot4, dot2), interp_1D(fx, dot3, dot1));
         return value;
    }
};

static struct Chunk *chunk_VBOs[VBO_SIZE] = {0};
static int now;

struct Chunk {
    Block blocks[CHUNK_WIDTH][CHUNK_HEIGHT][CHUNK_DEPTH];
    // Adjacent chunks
    struct Chunk *left, *right, *bottom, *top, *back, *front;
    
    GLuint vboVertices;
    
    int elementCount;
    int vboIndex;
    time_t lastUpdated;
    
    bool changed;
    bool initialized;
    bool generated;
    Vec3i position;
    
    Chunk(int px, int py, int pz) {
        memset(blocks, 0, sizeof(blocks));
        left = right = bottom = top = back = front = 0;
        
        this->position.x = px;
        this->position.y = py;
        this->position.z = pz;
        
        this->lastUpdated = now;
        this->elementCount = 0;
        this->vboIndex = 0;
        this->changed = true;
        this->initialized = false;
        this->generated = false;
    }
    
    Chunk() : Chunk(0, 0, 0) {}
    
    
    void dispose() {
        glDeleteBuffers(1, &this->vboVertices);
    }
    
    Block get_block(int x, int y, int z) {
        if (x < 0) return this->left ? this->left->blocks[x + CHUNK_WIDTH][y][z] : Block();
        if (x >= CHUNK_WIDTH) return this->right ? this->right->blocks[x - CHUNK_WIDTH][y][z] : Block();
        
        if (y < 0) return this->bottom ? this->bottom->blocks[x][y + CHUNK_HEIGHT][z] : Block();
        if (y >= CHUNK_HEIGHT) return this->top ? this->top->blocks[x][y - CHUNK_HEIGHT][z] : Block();
        
        if (z < 0) return this->back ? this->back->blocks[x][y][z + CHUNK_DEPTH] : Block();
        if (z >= CHUNK_DEPTH) return this->front ? this->front->blocks[x][y][z - CHUNK_DEPTH] : Block();
        
        // If inside the chunk, just return the block normally
        return this->blocks[x][y][z];
    }
    BlockTypes get_type(int x, int y, int z) {
        if (x < 0) return this->left ? this->left->blocks[x + CHUNK_WIDTH][y][z].type : BlockTypes::air;
        if (x >= CHUNK_WIDTH) return this->right ? this->right->blocks[x - CHUNK_WIDTH][y][z].type : BlockTypes::air;
        
        if (y < 0) return this->bottom ? this->bottom->blocks[x][y + CHUNK_HEIGHT][z].type : BlockTypes::air;
        if (y >= CHUNK_HEIGHT) return this->top ? this->top->blocks[x][y - CHUNK_HEIGHT][z].type : BlockTypes::air;
        
        if (z < 0) return this->back ? this->back->blocks[x][y][z + CHUNK_DEPTH].type : BlockTypes::air;
        if (z >= CHUNK_DEPTH) return this->front ? this->front->blocks[x][y][z - CHUNK_DEPTH].type : BlockTypes::air;
        
        // If inside the chunk, just return the type normally
        return this->blocks[x][y][z].type;
    }
    
    // Sets the block with its position being a 3-byte vector (-256 <= value <= 256)
    void set_block(int x, int y, int z, BlockTypes type) {
        Block b = Block(x, y, z, type);
        
        if (x < 0) {
            if (this->left) left->set_block(x + CHUNK_WIDTH, y, z, type);
            return;
        }
        if (x >= CHUNK_WIDTH) {
            if (this->right) right->set_block(x - CHUNK_WIDTH, y, z, type);
            return;
        }
        
        if (y < 0) {
            if (this->bottom) bottom->set_block(x, y + CHUNK_HEIGHT, z, type);
            return;
        }
        if (y >= CHUNK_HEIGHT) {
            if (this->top) top->set_block(x, y - CHUNK_HEIGHT, z, type);
            return;
        }
        
        if (z < 0) {
            if (this->back) back->set_block(x, y, z + CHUNK_DEPTH, type);
            return;
        }
        if (z >= CHUNK_DEPTH) {
            if (this->front) front->set_block(x, y, z - CHUNK_DEPTH, type);
            return;
        }
        
        this->blocks[x][y][z] = b;
        this->changed = true;
        
        if (x == 0 && this->left) left->changed = true;
        if (x == CHUNK_WIDTH - 1 && this->right) right->changed = true;
        
        if (y == 0 && this->bottom) bottom->changed = true;
        if (y == CHUNK_HEIGHT - 1 && this->top) top->changed = true;
        
        if (z == 0 && this->back) back->changed = true;
        if (z == CHUNK_DEPTH - 1 && this->front) front->changed = true;
    }
    
    void set_block_gen(int x, int y, int z, BlockTypes type) {
        Block b = Block(x, y, z, type);
        
        if (x < 0) {
            if (this->left) left->set_block_gen(x + CHUNK_WIDTH, y, z, type);
            return;
        }
        if (x >= CHUNK_WIDTH) {
            if (this->right) right->set_block_gen(x - CHUNK_WIDTH, y, z, type);
            return;
        }
        
        if (y < 0) {
            if (this->bottom) bottom->set_block_gen(x, y + CHUNK_HEIGHT, z, type);
            return;
        }
        if (y >= CHUNK_HEIGHT) {
            if (this->top) top->set_block_gen(x, y - CHUNK_HEIGHT, z, type);
            return;
        }
        
        if (z < 0) {
            if (this->back) back->set_block_gen(x, y, z + CHUNK_DEPTH, type);
            return;
        }
        if (z >= CHUNK_DEPTH) {
            if (this->front) front->set_block_gen(x, y, z - CHUNK_DEPTH, type);
            return;
        }
        
        this->blocks[x][y][z] = b;
    }
    
    bool is_type_obs(int x1, int y1, int z1, int x2, int y2, int z2) {
        uint8_t b1 = get_index_from_type(get_type(x1, y1, z1));
        uint8_t b2 = get_index_from_type(get_type(x2, y2, z2));
        
        if (!b1) {
            return true;
        }
        if (transparency[b2] == 1) {
            return false;
        }
        if (!transparency[b2]) {
            return true;
        }
        
        return transparency[b2] == transparency[b1];
    }
    
    float perlin_2D(float x, float y, float scale, float magnitude) {
        float s = float(Perlin::noise(x / scale, y / scale) * magnitude);
        
        return s;
    }
    float perlin_2D(float x, float y, int octaves, float persistence) {
        float sum = 0.0f, stregth = 1.0f, scale = 1.0f;
        
        for (int i = 0; i < octaves; i++) {
             sum += stregth * Perlin::noise(x / scale, y / scale);
             scale *= 2.0f;
             stregth *= persistence;
        }
        
        return sum;
    }
    
    void noise() {
        if (this->generated) {
            return; 
        } else {
            this->generated = true;
        }
        
        for (int x = 0; x < CHUNK_WIDTH; x++) {
            for (int z = 0; z < CHUNK_DEPTH; z++) {
                 float m = this->perlin_2D((x + position.x * CHUNK_WIDTH + COLLECTION_WIDTH * CHUNK_WIDTH / 2) / 30.0, (z + position.z * CHUNK_DEPTH + COLLECTION_DEPTH * CHUNK_DEPTH / 2) / 30.0, 4, 0.8) * 2;
                 uint8_t h = m * 2;
                 
                 
                 for (int y = 0; y < CHUNK_HEIGHT; y++) {
                     // Dirt terrain
                     if (y + position.y * CHUNK_HEIGHT <= h && y >= 0) {
                         this->set_block_gen(x, y - 1, z, BlockTypes::dirt);  
                     }
                 }
                 /*
                 for (int y = 0; y < CHUNK_HEIGHT; y++) {
                     if (y + position.y * CHUNK_HEIGHT < 2) {
                         if (this->get_type(x, y, z) == BlockTypes::air) {
                             this->set_block_gen(x, y, z, BlockTypes::water);
                         }
                     }
                 }
                 */
                 
                 for (int y = 0; y < CHUNK_HEIGHT; y++) {
                     if (y + position.y * CHUNK_HEIGHT < -5) {
                         // Stone layer
                         this->set_block_gen(x, y, z, BlockTypes::stone);
                     }
                 }
                 
                 BlockTypes b = this->get_type(x, h - 1, z);
                 if (b == BlockTypes::dirt || b == BlockTypes::grass) {
                     // A layer of grass blocks on top of the dirt blocks
                     this->set_block_gen(x, h, z, BlockTypes::grass);
                      
                     // Trees
                     if (rand() % 300 == 1) {
                         this->tree(x, z, h);
                     }
                 }
            }
        }
        this->changed = true;
    }
    void tree(int x, int y, int height) {
        // Wooden trunk as base        
        int trunkHeight = rand() % 7 + 5;
        for (int h = 1; h < trunkHeight + 1; h++) {
             // Don't generate trees that are stuck in the ground
             if (height + h + position.y * CHUNK_HEIGHT >= height) {
                 this->set_block_gen(x, height + h, y, BlockTypes::woodLog);
             }
        }
        
        // Leaves
        if (height + position.y * CHUNK_HEIGHT >= height) {
            for (int mx = -3; mx < 3; mx++) {
                 for (int my = -3; my < 3; my++) {
                      for (int mz = -3; mz < 3; mz++) {
                           if (pow(mx, 2) + pow(my, 2) + pow(mz, 2) < 12 + (rand() & 1) && get_type(mx, my + height + trunkHeight, mz) == BlockTypes::air) {
                               int tx = x + mx;
                               int ty = height + trunkHeight + my;
                               int tz = y + mz;
                           
                               this->set_block_gen(tx, ty, tz, BlockTypes::leaves);
                           }
                      }
                  }
             }
        }
    }
    
    
    
    void update() {
        bool visible = false;        
        Vec4b vertices[COMBINED_SIZE * 18];
        int i = 0;
        
        // -x
        for (int x = CHUNK_WIDTH - 1; x >= 0; x--) {
             for (int y = 0; y < CHUNK_HEIGHT; y++) {
                  for (int z = 0; z < CHUNK_DEPTH; z++) {
                      if (is_type_obs(x, y, z, x - 1, y, z)) {
                          visible = false;
                          continue;
                      }
                      BlockTypes type = get_type(x, y, z);
                      uint8_t top, side;
                      
                      // Get the index
                      top = side = get_index_from_type(type);
                      // Also change the type of a block's face if necessary
                      if (top == 1) {
                          side = 5;
                      }
                      
                      if (visible && z != 0 && type == get_type(x, y, z - 1)) {
                          vertices[i - 5] = Vec4b(x, y, z + 1, side);
                          vertices[i - 2] = Vec4b(x, y, z + 1, side);
                          vertices[i - 1] = Vec4b(x, y + 1, z + 1, side);
                      } else {
                          vertices[i++] = Vec4b(x, y, z, side);
                          vertices[i++] = Vec4b(x, y, z + 1, side);
                          vertices[i++] = Vec4b(x, y + 1, z, side);
                          vertices[i++] = Vec4b(x, y + 1, z, side);
                          vertices[i++] = Vec4b(x, y, z + 1, side);
                          vertices[i++] = Vec4b(x, y + 1, z + 1, side);
                      } 
                      visible = true; 
                 }
            }
       }
       
       // +x  
       for (int x = 0; x < CHUNK_WIDTH; x++) {
            for (int y = 0; y < CHUNK_HEIGHT; y++) {
                 for (int z = 0; z < CHUNK_DEPTH; z++) {
                      if (is_type_obs(x, y, z, x + 1, y, z)) {
                          visible = false;
                          continue;
                      }
                      BlockTypes type = get_type(x, y, z);
                      uint8_t top, side;
                      
                      top = side = get_index_from_type(type);
                      if (top == 1) {
                          side = 5;
                      }
                      
                      uint8_t offset = 16;
                      if (visible && z != 0 && type == get_type(x, y, z - 1)) {
                          vertices[i - 4] = Vec4b(x + 1, y, z + 1, side + offset);
                          vertices[i - 2] = Vec4b(x + 1, y + 1, z + 1, side + offset);
                          vertices[i - 1] = Vec4b(x + 1, y, z + 1, side + offset);
                      } else {
                          vertices[i++] = Vec4b(x + 1, y, z, side + offset);
                          vertices[i++] = Vec4b(x + 1, y + 1, z, side + offset);
                          vertices[i++] = Vec4b(x + 1, y, z + 1, side + offset);
                          vertices[i++] = Vec4b(x + 1, y + 1, z, side + offset);
                          vertices[i++] = Vec4b(x + 1, y + 1, z + 1, side + offset);
                          vertices[i++] = Vec4b(x + 1, y, z + 1, side + offset);
                      }
                      visible = true;
                 }
            }
       }
       
       // -y
       for (int x = 0; x < CHUNK_WIDTH; x++) {
             for (int y = CHUNK_HEIGHT - 1; y >= 0; y--) {
                  for (int z = 0; z < CHUNK_DEPTH; z++) {
                      if (is_type_obs(x, y, z, x, y - 1, z)) {
                          visible = false;
                          continue;
                      }
                      BlockTypes type = get_type(x, y, z);
                      uint8_t top, bottom;
                      
                      top = bottom = get_index_from_type(type);
                      if (top == 1) {
                          bottom = 4;
                      } else if (top == 2) {
                          bottom = 8;
                      }
                      
                      
                      uint8_t offset = 32;
                      if (visible && z != 0 && type == get_type(x, y, z - 1)) {
                          vertices[i - 4] = Vec4b(x, y, z + 1, bottom + offset);
                          vertices[i - 2] = Vec4b(x + 1, y, z + 1, bottom + offset);
                          vertices[i - 1] = Vec4b(x, y, z + 1, bottom + offset);
                      } else {
                          vertices[i++] = Vec4b(x, y, z, bottom + offset);
                          vertices[i++] = Vec4b(x + 1, y, z, bottom + offset);
                          vertices[i++] = Vec4b(x, y, z + 1, bottom + offset);
                          vertices[i++] = Vec4b(x + 1, y, z, bottom + offset);
                          vertices[i++] = Vec4b(x + 1, y, z + 1, bottom + offset);
                          vertices[i++] = Vec4b(x, y, z + 1, bottom + offset);
                      } 
                      visible = true; 
                 }
            }
       }
       
       // +y
       for (int x = 0; x < CHUNK_WIDTH; x++) {
             for (int y = 0; y < CHUNK_HEIGHT; y++) {
                  for (int z = 0; z < CHUNK_DEPTH; z++) {
                      if (is_type_obs(x, y, z, x, y + 1, z)) {
                          visible = false;
                          continue;
                      }
                      BlockTypes type = get_type(x, y, z);
                      uint8_t top = get_index_from_type(type);
                      
                      if (type == BlockTypes::woodLog) {
                          top = 8;
                      }
                      
                      uint8_t offset = 48;
                      if (visible && z != 0 && type == get_type(x, y, z - 1)) {
                          vertices[i - 5] = Vec4b(x, y + 1, z + 1, top + offset);
                          vertices[i - 2] = Vec4b(x, y + 1, z + 1, top + offset);
                          vertices[i - 1] = Vec4b(x + 1, y + 1, z + 1, top + offset);
                      } else {
                          vertices[i++] = Vec4b(x, y + 1, z, top + offset);
                          vertices[i++] = Vec4b(x, y + 1, z + 1, top + offset);
                          vertices[i++] = Vec4b(x + 1, y + 1, z, top + offset);
                          vertices[i++] = Vec4b(x + 1, y + 1, z, top + offset);
                          vertices[i++] = Vec4b(x, y + 1, z + 1, top + offset);
                          vertices[i++] = Vec4b(x + 1, y + 1, z + 1, top + offset);
                      }
                      visible = true;
                 }
            }
       }
       
       // -z
       for (int x = 0; x < CHUNK_WIDTH; x++) {
             for (int z = CHUNK_DEPTH - 1; z >= 0; z--) {
                  for (int y = 0; y < CHUNK_HEIGHT; y++) {
                      if (is_type_obs(x, y, z, x, y, z - 1)) {
                          visible = false;
                          continue;
                      }
                      BlockTypes type = get_type(x, y, z);
                      uint8_t top, side;
                      
                      top = side = get_index_from_type(type);
                      if (top == 1) {
                          side = 5;
                      }
                      
                      uint8_t offset = 64;
                      if (visible && y != 0 && type == get_type(x, y - 1, z)) {
                          vertices[i - 5] = Vec4b(x, y + 1, z, side + offset);
                          vertices[i - 3] = Vec4b(x, y + 1, z, side + offset);
                          vertices[i - 2] = Vec4b(x + 1, y + 1, z, side + offset);
                      } else {
                          vertices[i++] = Vec4b(x, y, z, side + offset);
                          vertices[i++] = Vec4b(x, y + 1, z, side + offset);
                          vertices[i++] = Vec4b(x + 1, y, z, side + offset);
                          vertices[i++] = Vec4b(x, y + 1, z, side + offset);
                          vertices[i++] = Vec4b(x + 1, y + 1, z, side + offset);
                          vertices[i++] = Vec4b(x + 1, y, z, side + offset);
                      }
                      visible = true;
                  }
             }
        }
         
        // +z
        for (int x = 0; x < CHUNK_WIDTH; x++) {
             for (int z = 0; z < CHUNK_DEPTH; z++) {
                  for (int y = 0; y < CHUNK_HEIGHT; y++) {
                      if (is_type_obs(x, y, z, x, y, z + 1)) {
                          visible = false;
                          continue;
                      }
                      BlockTypes type = get_type(x, y, z);
                      uint8_t top, side;
                      
                      top = side = get_index_from_type(type);
                      if (top == 1) {
                          side = 5;
                      }
                      
                      uint8_t offset = 80;
                      if (visible && y != 0 && type == get_type(x, y - 1, z)) {
                          vertices[i - 4] = Vec4b(x, y + 1, z + 1, side + offset);
                          vertices[i - 3] = Vec4b(x, y + 1, z + 1, side + offset);
                          vertices[i - 1] = Vec4b(x + 1, y + 1, z + 1, side + offset);
                      } else {
                          vertices[i++] = Vec4b(x, y, z + 1, side + offset);
                          vertices[i++] = Vec4b(x + 1, y, z + 1, side + offset);
                          vertices[i++] = Vec4b(x, y + 1, z + 1, side + offset);
                          vertices[i++] = Vec4b(x, y + 1, z + 1, side + offset);
                          vertices[i++] = Vec4b(x + 1, y, z + 1, side + offset);
                          vertices[i++] = Vec4b(x + 1, y + 1, z + 1, side + offset);
                      }
                      visible = true;
                  }
             }
        }
        
        this->changed = false;
        this->elementCount = i;
        
        if (!this->elementCount)
            return;
        
        if (chunk_VBOs[vboIndex] != this) {
            int u = 0;
            for (int i = 0; i < VBO_SIZE; i++) {
                if (!chunk_VBOs[i]) {
                    u = i;
                    break;
                }
                if (chunk_VBOs[i]->lastUpdated < chunk_VBOs[u]->lastUpdated) {
                    u = i;
                }
            }
            
            if (!chunk_VBOs[u]) {
                glGenBuffers(1, &this->vboVertices);
            } else {
                this->vboVertices = chunk_VBOs[u]->vboVertices;
                chunk_VBOs[u]->changed = true;
            }
            
            this->vboIndex = u;
            chunk_VBOs[this->vboIndex] = this;
        }
        
        glBindBuffer(GL_ARRAY_BUFFER, this->vboVertices);
        glBufferData(GL_ARRAY_BUFFER, i * sizeof *vertices, vertices, GL_STATIC_DRAW);
    }
    
    void render() {
        if (this->changed) {
            this->update();
        }
        this->lastUpdated = now;
        
        if (!this->elementCount) return;
        
        Shaders::get().use("testShader");
        
        Shaders::get().find("testShader")->set_uniform_mat4("view", Projection::viewMat);
        Shaders::get().find("testShader")->set_uniform_mat4("projection", Projection::projMat);
        
        GLint position = Shaders::get().find("testShader")->attribute_location("position");
         
        glBindBuffer(GL_ARRAY_BUFFER, this->vboVertices);
        glVertexAttribPointer(position, 4, GL_BYTE, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(position);
        
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei) this->elementCount);
           
        glDisableVertexAttribArray(position);     
    }
};

struct ChunkCollection {
    Chunk *chunks[COLLECTION_WIDTH][COLLECTION_HEIGHT][COLLECTION_DEPTH];
    Texture *blockTexture;
    
    ChunkCollection() {
        this->blockTexture = new Texture("textures.png");
        
        // Add the chunks
        for (int x = 0; x < COLLECTION_WIDTH; x++) {
            for (int y = 0; y < COLLECTION_HEIGHT; y++) {
                for (int z = 0; z < COLLECTION_DEPTH; z++) {
                    chunks[x][y][z] = new Chunk(x - COLLECTION_WIDTH / 2, y - COLLECTION_HEIGHT / 2, z - COLLECTION_DEPTH / 2);
                }
            }
        }
        
        // Set up the adjacent chunks
        for (int x = 0; x < COLLECTION_WIDTH; x++) {
            for (int y = 0; y < COLLECTION_HEIGHT; y++) {
                for (int z = 0; z < COLLECTION_DEPTH; z++) {
                     // X-axis
                     if (x > 0) {
                         chunks[x][y][z]->left = chunks[x - 1][y][z];
                     }
                     if (x < COLLECTION_WIDTH - 1) {
                         chunks[x][y][z]->right = chunks[x + 1][y][z];
                     }
                     
                     // Y-axis
                     if (y > 0) {
                         chunks[x][y][z]->bottom = chunks[x][y - 1][z];
                     }
                     if (y < COLLECTION_HEIGHT - 1) {
                         chunks[x][y][z]->top = chunks[x][y + 1][z];
                     }
                     
                     // Z-axis
                     if (z > 0) {
                         chunks[x][y][z]->back = chunks[x][y][z - 1];
                     }
                     if (z < COLLECTION_DEPTH - 1) {
                         chunks[x][y][z]->front = chunks[x][y][z + 1];
                     }
                }
            }
        }
        
        this->blockTexture->load();
    };
    
    BlockTypes get_type(int x, int y, int z) {
        int mx = (x + CHUNK_WIDTH * (COLLECTION_WIDTH / 2)) / CHUNK_WIDTH;
        int my = (y + CHUNK_HEIGHT * (COLLECTION_HEIGHT / 2)) / CHUNK_HEIGHT;
        int mz = (z + CHUNK_DEPTH * (COLLECTION_DEPTH / 2)) / CHUNK_DEPTH;
        
        if (mx < 0 || mx >= COLLECTION_WIDTH ||
            my < 0 || my >= COLLECTION_HEIGHT ||
            mz < 0 || mz >= COLLECTION_DEPTH) return BlockTypes::air;
        
        return chunks[mx][my][mz]->get_type(
            x & (CHUNK_WIDTH - 1),
            y & (CHUNK_HEIGHT - 1),
            z & (CHUNK_DEPTH - 1)   
        );
    }
    Block get_block(int x, int y, int z) {
        int mx = (x + CHUNK_WIDTH * (COLLECTION_WIDTH / 2)) / CHUNK_WIDTH;
        int my = (y + CHUNK_HEIGHT * (COLLECTION_HEIGHT / 2)) / CHUNK_HEIGHT;
        int mz = (z + CHUNK_DEPTH * (COLLECTION_DEPTH / 2)) / CHUNK_DEPTH;
        
        if (mx < 0 || mx >= COLLECTION_WIDTH ||
            my < 0 || my >= COLLECTION_HEIGHT ||
            mz < 0 || mz >= COLLECTION_DEPTH) return Block();
        
        return chunks[mx][my][mz]->get_block(
            x & (CHUNK_WIDTH - 1),
            y & (CHUNK_HEIGHT - 1),
            z & (CHUNK_DEPTH - 1)   
        );
    }
    void set_block(int x, int y, int z, BlockTypes type) {
        int mx = (x + CHUNK_WIDTH * (COLLECTION_WIDTH / 2)) / CHUNK_WIDTH;
        int my = (y + CHUNK_HEIGHT * (COLLECTION_HEIGHT / 2)) / CHUNK_HEIGHT;
        int mz = (z + CHUNK_DEPTH * (COLLECTION_DEPTH / 2)) / CHUNK_DEPTH;
        
        if (mx < 0 || mx >= COLLECTION_WIDTH ||
            my < 0 || my >= COLLECTION_HEIGHT ||
            mz < 0 || mz >= COLLECTION_DEPTH) return;
        
        chunks[mx][my][mz]->set_block(
            x & (CHUNK_WIDTH - 1),
            y & (CHUNK_HEIGHT - 1),
            z & (CHUNK_DEPTH - 1),
            type   
        );
    }
    
    // A naive way of getting the height relative to the bottom of the level
    int get_height(int x, int z) {
        int cy = COLLECTION_HEIGHT * CHUNK_HEIGHT / 2;
        
        int height = -cy;
        for (int y = -cy; y < cy; y++) {
            if ((this->get_type(x, y, z) != BlockTypes::air || y > height)) {
                height = y; 
            }
        }
        
        return height;
    }
    
    void render() {
        Shaders::get().use("testShader");
        
        
        this->blockTexture->draw();
        
        float mDist = 1.0 / 0.0;
        int mx = -1.0;
        int my = -1.0;
        int mz = -1.0;
        for (int x = 0; x < COLLECTION_WIDTH; x++) {
            for (int y = 0; y < COLLECTION_HEIGHT; y++) {
                for (int z = 0; z < COLLECTION_DEPTH; z++) {
                    Chunk *c = chunks[x][y][z];
                    if (c) {
                        // The modelview matrix takes into account the camera's rotation
                        Mat4x4 model, modelView;
                        model.set_translation(c->position.x * CHUNK_WIDTH, c->position.y * CHUNK_HEIGHT, c->position.z * CHUNK_DEPTH);
                        modelView = model.multiply(Projection::viewMat);
                        
                        Vec4f chunkCenter = { CHUNK_WIDTH / 2, CHUNK_HEIGHT / 2, CHUNK_DEPTH / 2, 1};
                        Vec4f pos = modelView.multiply_vector(chunkCenter);
                        Vec4f camera = { Projection::cameraPosition.x, Projection::cameraPosition.y, Projection::cameraPosition.z, 1 };
                        
                        float dst = camera.dst(pos);
                        pos.x /= pos.w;
                        pos.y /= pos.w;
                        pos.z /= pos.w;
                        
                        if (pos.z < -CHUNK_HEIGHT / 2) continue;
                        
                        if (fabsf(pos.x) > 1 + fabsf(CHUNK_HEIGHT * 2 / pos.w) ||
                            fabsf(pos.y) > 1 + fabsf(CHUNK_HEIGHT * 2 / pos.w)) continue;
                            
                        
                        if (!c->initialized) {
                            if (mx < 0 || dst < mDist) {
                                mDist = dst;
                                mx = x;
                                my = y;
                                mz = z;
                            }
                            continue;
                        }
                        
                        Shaders::get().find("testShader")->set_uniform_mat4("model", model);
    
                        c->render();
                    }
                }
            }
        }
        
        if (mx >= 0) {
            chunks[mx][my][mz]->noise();
            
            if (chunks[mx][my][mz]->left) chunks[mx][my][mz]->left->noise();
            if (chunks[mx][my][mz]->right) chunks[mx][my][mz]->right->noise();
            
            if (chunks[mx][my][mz]->bottom) chunks[mx][my][mz]->bottom->noise();
            if (chunks[mx][my][mz]->top) chunks[mx][my][mz]->top->noise();
            
            if (chunks[mx][my][mz]->back) chunks[mx][my][mz]->back->noise();
            if (chunks[mx][my][mz]->front) chunks[mx][my][mz]->front->noise();
            
            chunks[mx][my][mz]->initialized = true;
        }
    }
    
    void dispose() {
        for (int x = 0; x < COLLECTION_WIDTH; x++) {
            for (int y = 0; y < COLLECTION_HEIGHT; y++) {
                for (int z = 0; z < COLLECTION_DEPTH; z++) {
                    chunks[x][y][z]->dispose();
                }
            }
        }
        this->blockTexture->clear();
    }
};
ChunkCollection *level;

class Ray {
    public:
       Vec3f startPosition, endPosition;
       float rangeLimit;
       
       Ray(float rangeLimit) {
           this->startPosition.set_zero();
           this->endPosition.set_zero();
           
           this->rangeLimit = rangeLimit;
       }
       
       void set_start_pos(Vec3f to) {
           this->startPosition.x = to.x;
           this->startPosition.y = to.y;
           this->startPosition.z = to.z;
       }
       
       void set_end_pos(Vec3f to) {
           this->endPosition.x = to.x;
           this->endPosition.y = to.y;
           this->endPosition.z = to.z;
       }
       
       // Returns the intersection point of the ray with the specific voxel
       Vec3i update_DDA(bool placing) {
           // Also refer to OLC's implementation of the DDA algorithm
           Vec3f direction = { this->endPosition.x - this->startPosition.x,
                               this->endPosition.y - this->startPosition.y,
                               this->endPosition.z - this->startPosition.z };
           direction.norm();
           
           Vec3f stepSize = { fabsf(1.0f / direction.x), fabsf(1.0f / direction.y), fabsf(1.0f / direction.z) };
           Vec3i step;
           
           Vec3f lengths;
           
           Vec3i voxelCheck = Vec3i(startPosition.x, startPosition.y, startPosition.z);
           
           // Left, right
           if (direction.x < 0) {
               step.x = -1;
               lengths.x = (startPosition.x - (float) voxelCheck.x) * stepSize.x;
           } else {
               step.x = 1;
               lengths.x = (float(voxelCheck.x + 1) - startPosition.x) * stepSize.x;
           }
           
           // Down, up
           if (direction.y < 0) {
               step.y = -1;
               lengths.y = (startPosition.y - (float) voxelCheck.y) * stepSize.y;
           } else {
               step.y = 1;
               lengths.y = (float(voxelCheck.y + 1) - startPosition.y) * stepSize.y;
           }
           
           // Back, front
           if (direction.z < 0) {
               step.z = -1;
               lengths.z = (startPosition.z - (float) voxelCheck.z) * stepSize.z;
           } else {
               step.z = 1;
               lengths.z = (float(voxelCheck.z + 1) - startPosition.z) * stepSize.z;
           }
           
           bool intersection = false;
           float distance = 0.0f;
           while (!intersection && distance < rangeLimit) {
                if (lengths.x < lengths.y && lengths.x < lengths.z) {
                    // Going in the X axis
                    voxelCheck.x += step.x;
                    distance = lengths.x;
                    lengths.x += stepSize.x;
                } else if (lengths.y < lengths.z) {
                    // Going in the Y axis
                    voxelCheck.y += step.y;
                    distance = lengths.y;
                    lengths.y += stepSize.y;
                } else {
                    // Going in the Z axis
                    voxelCheck.z += step.z;
                    distance = lengths.z; 
                    lengths.z += stepSize.z;
                }
                
                if (placing) {
                    // Redo the step calculations, now tring to check intersection with a ray that is a full step further.
                    if (lengths.x < lengths.y && lengths.x < lengths.z) {
                        if (level->get_type(voxelCheck.x, voxelCheck.y, voxelCheck.z) == BlockTypes::air &&
                            level->get_type(voxelCheck.x + step.x, voxelCheck.y, voxelCheck.z) != BlockTypes::air) {
                                
                            intersection = true;
                            return voxelCheck;
                        }
                    } else if (lengths.y < lengths.z) {
                        if (level->get_type(voxelCheck.x, voxelCheck.y, voxelCheck.z) == BlockTypes::air &&
                            level->get_type(voxelCheck.x, voxelCheck.y + step.y, voxelCheck.z) != BlockTypes::air) {
                                
                            intersection = true;
                            return voxelCheck;
                        }
                    } else {
                        if (level->get_type(voxelCheck.x, voxelCheck.y, voxelCheck.z) == BlockTypes::air &&
                            level->get_type(voxelCheck.x, voxelCheck.y, voxelCheck.z + step.z) != BlockTypes::air) {
                                
                            intersection = true;
                            return voxelCheck;
                        }
                    }
                } else {
                    if (level->get_type(voxelCheck.x, voxelCheck.y, voxelCheck.z) != BlockTypes::air) {
                        intersection = true;
                        return voxelCheck;
                    }
                }
           }
           return Vec3i();
       }
};


struct Player {
    Vec3f position;
    Vec3f vel;
    Vec3f acceleration;
    
    float speed, jumpForce, resistance;
    
    float rotationX, rotationY;
    bool collidingGround;
    
    AABB aabb;
    Chunk *on;
    Vec3i lookingAt;
    Ray *ray;
    GLuint outlineVBO;
    BlockTypes selectedBlock;
    
    Player() {
        position.set_zero();
        vel.set_zero();
        acceleration.set_zero();
        
        speed = 50.0f;
        this->jumpForce = 7.0f;
        resistance = 0.85f;
        rotationX = rotationY = 0.0f;
        
        collidingGround = false;
        this->on = 0;
        this->selectedBlock = BlockTypes::dirt;
        this->place(0, 0);
        this->ray = new Ray(5.0f);
       
        float outline[24 * 3] = {
            -1.0, -1.0, -1.0,
            1.0, -1.0, -1.0,
            -1.0, 1.0, -1.0,
            1.0, 1.0, -1.0,
            
            -1.0, -1.0, 1.0,
            1.0, -1.0, 1.0,
            -1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            
            -1.0, -1.0, -1.0,
            -1.0, 1.0, -1.0,
            1.0, -1.0, -1.0,
            1.0, 1.0, -1.0,
            
            -1.0, -1.0, 1.0,
            -1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, -1.0, 1.0,
            
            -1.0, -1.0, -1.0,
            -1.0, -1.0, 1.0,
            1.0, -1.0, -1.0,
            1.0, -1.0, 1.0,
            
            -1.0, 1.0, -1.0,
            -1.0, 1.0, 1.0,
            1.0, 1.0, -1.0,
            1.0, 1.0, 1.0,
        };
        for (int i = 0; i < 24 * 3; i++) {
            outline[i] *= 0.5;
        }
        for (int i = 0; i < 24 * 3; i++) {
            outline[i] += 0.5;
        }
        glGenBuffers(1, &this->outlineVBO);
        glBindBuffer(GL_ARRAY_BUFFER, this->outlineVBO);
        glBufferData(GL_ARRAY_BUFFER, 30 * 3 * sizeof(float), outline, GL_STATIC_DRAW);
    }
    bool collide_AA(AABB a, AABB b) {
         return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
                (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
                (a.min.z <= b.max.z && a.max.z >= b.min.z);
    }
    
    AABB get_AABB() {
         return aabb;
    }
    
    void place(int x, int z) {
        int tx = x;
        int ty = level->get_height(x, z);
        int tz = z;
        
        this->position.x = tx;
        this->position.y = ty;
        this->position.z = tz;
    }
    
    // Tiled colllision detection
    void update(float timeTook) {
        acceleration.x = -vel.x * resistance;
        acceleration.y = -vel.y * resistance - 9.8f;
        acceleration.z = -vel.z * resistance;
        
        vel.x += acceleration.x * timeTook;
        vel.y += acceleration.y * timeTook;
        vel.z += acceleration.z * timeTook;
        
        // Damping
        vel.x *= 0.85;
        vel.z *= 0.85;
        
        position.x += vel.x * timeTook;
        position.y += vel.y * timeTook;
        position.z += vel.z * timeTook;
        
        this->on = this->chunk_on(round(position.x), round(position.y), round(position.z));
       
        update_AABB();
        if (this->on) {
            collision_detection();
        }
        update_AABB();
        
        

        Projection::cameraPosition.x = position.x;
        Projection::cameraPosition.y = position.y;
        Projection::cameraPosition.z = position.z;
        
        Vec3f lookAt = { Projection::cameraPosition.x + (cos(rotationX) * cos(rotationY)) * 20.0f,
                         Projection::cameraPosition.y + sin(rotationY) * 20.0f,
                         Projection::cameraPosition.z + (sin(rotationX) * cos(rotationY)) * 20.0f };
        
        this->ray->set_start_pos(this->position);
        this->ray->set_end_pos(lookAt);
        
        Vec3i dda = this->ray->update_DDA(false);   
        
        this->lookingAt = dda;
    }
    
    void collision_detection() {
        // Get the scaled chunk position
        Vec3f chunkPos = {
            (float)on->position.x * CHUNK_WIDTH,
            (float)on->position.y * CHUNK_HEIGHT,
            (float)on->position.z * CHUNK_DEPTH
        };
        
        // Bottom-top collisions
            Block bottom = level->get_block(round(position.x), round(position.y - 2), round(position.z));
            if (bottom.type != BlockTypes::air) {
                if (collide_AA(this->get_AABB(), bottom.get_AABB(bottom.position).add(chunkPos)) && vel.y < 0) {
                    vel.y = 0;
                    position.y = bottom.position.y + chunkPos.y + 2.0f;
                    
                    collidingGround = true;
                }
            } else {
                collidingGround = false;
            }
            Block top = level->get_block(round(position.x), round(position.y + 1), round(position.z));  
            if (top.type != BlockTypes::air) {
                if (collide_AA(this->get_AABB(), top.get_AABB(top.position).add(chunkPos))) {
                    vel.y = 0;
                    position.y = top.position.y + chunkPos.y - 1.0f;
                }
            }
        
        // Left-right collisions
            Block left = level->get_block(round(position.x - 1), round(position.y - 1), round(position.z));  
            if (left.type != BlockTypes::air) {
                if (collide_AA(this->get_AABB(), left.get_AABB(left.position).add(chunkPos))) {
                    vel.x = 0;
                    position.x = left.position.x + chunkPos.x + 1.0f;
                }
            }  
        
            Block left2 = level->get_block(round(position.x - 1), round(position.y), round(position.z));  
            if (left2.type != BlockTypes::air) {
                if (collide_AA(this->get_AABB(), left2.get_AABB(left2.position).add(chunkPos))) {
                    vel.x = 0;
                    position.x = left2.position.x + chunkPos.x + 1.0f;
                }
            }
            Block right = level->get_block(round(position.x + 1), round(position.y - 1), round(position.z));  
            if (right.type != BlockTypes::air) {
                if (collide_AA(this->get_AABB(), right.get_AABB(right.position).add(chunkPos))) {
                    vel.x = 0;
                    position.x = right.position.x + chunkPos.x - 1.0f;
                }
            }
              
            Block right2 = level->get_block(round(position.x + 1), round(position.y), round(position.z));  
            if (right2.type != BlockTypes::air) {
                if (collide_AA(this->get_AABB(), right2.get_AABB(right2.position).add(chunkPos))) {
                    vel.x = 0;
                    position.x = right2.position.x + chunkPos.x - 1.0f;
                }
            }
        
        // Front-back collisions
            Block back = level->get_block(round(position.x), round(position.y - 1), round(position.z - 1));  
            if (back.type != BlockTypes::air) {
                if (collide_AA(this->get_AABB(), back.get_AABB(back.position).add(chunkPos))) {
                    vel.z = 0;
                    position.z = back.position.z + chunkPos.z + 1.0f;
                }
            }  
        
            Block back2 = level->get_block(round(position.x), round(position.y), round(position.z - 1));  
            if (back2.type != BlockTypes::air) {
                if (collide_AA(this->get_AABB(), back2.get_AABB(back2.position).add(chunkPos))) {
                    vel.z = 0;
                    position.z = back2.position.z + chunkPos.z + 1.0f;
                }
            }
            Block front = level->get_block(round(position.x), round(position.y - 1), round(position.z + 1));  
            if (front.type != BlockTypes::air) {
                if (collide_AA(this->get_AABB(), front.get_AABB(front.position).add(chunkPos))) {
                    vel.z = 0;
                    position.z = front.position.z + chunkPos.z - 1.0f;
                }
            }  
        
            Block front2 = level->get_block(round(position.x), round(position.y), round(position.z + 1));  
            if (front2.type != BlockTypes::air) {
                if (collide_AA(this->get_AABB(), front2.get_AABB(front2.position).add(chunkPos))) {
                    vel.z = 0;
                    position.z = front2.position.z + chunkPos.z - 1.0f;
                }
            }
    }
    
    Chunk *chunk_on(int x, int y, int z) {
        int px = (x + CHUNK_WIDTH * (COLLECTION_WIDTH / 2)) / CHUNK_WIDTH;
        int py = (y + CHUNK_HEIGHT * (COLLECTION_HEIGHT / 2)) / CHUNK_HEIGHT;
        int pz = (z + CHUNK_DEPTH * (COLLECTION_DEPTH / 2)) / CHUNK_DEPTH;
        
        if (px < 0 || px >= COLLECTION_WIDTH ||
            py < 0 || py >= COLLECTION_HEIGHT ||
            pz < 0 || pz >= COLLECTION_DEPTH) return 0;
        
        return level->chunks[px][py][pz];
    }
    
    void render_outline(float timePassed) {
        Vec3i r = this->lookingAt;
        
        // Draw an outline of the cube we're pointing at
        Shaders::get().use("overlayShader");
        
        Mat4x4 model;
        model.set_translation(lookingAt.x, lookingAt.y, lookingAt.z);
        Shaders::get().find("overlayShader")->set_uniform_float("u_time", timePassed);
       
        Shaders::get().find("overlayShader")->set_uniform_mat4("u_model", model);
        Shaders::get().find("overlayShader")->set_uniform_mat4("u_view", Projection::viewMat);
        Shaders::get().find("overlayShader")->set_uniform_mat4("u_projection", Projection::projMat);
            
        GLint position = Shaders::get().find("overlayShader")->attribute_location("a_pos");
        
        glBindBuffer(GL_ARRAY_BUFFER, this->outlineVBO);
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(position);
        
        glDrawArrays(GL_LINES, 0, 24);
           
        glDisableVertexAttribArray(position);
    }
    void set_selected(BlockTypes type) {
        this->selectedBlock = type;
    }
    
    void jump() {
        if (this->collidingGround) {
            this->vel.y += this->jumpForce;
            this->collidingGround = false;
        }
    }
    void update_AABB() {
        this->aabb.min = { position.x - 0.5f, position.y - 1.5f, position.z - 0.5f };
        this->aabb.max = { position.x + 0.5f, position.y + 0.5f, position.z + 0.5f };
    }
    
    void cleanup() {
        glDeleteBuffers(1, &this->outlineVBO);
    }
};
Player *player;
   
namespace UI {
    bool focused;
    
    class Element {
        public:
            // Center position
            Vec2f position;
            Vec2f scaling;
            Element() {
                this->texture = nullptr;
                this->selected = [&]() -> bool { return false; };
                
                this->vertices = {
                    // Vertices
                    -0.5f, -0.5f, 0.0f,
                    0.5f, -0.5f, 0.0f,
                    0.5f, 0.5f, 0.0f,
                    -0.5f, 0.5f, 0.0f
                };
                
                
                this->textureCoords = {
                    // Texture coordinates
                    0.0f, 0.0f,
                    1.0f, 0.0f,
                    1.0f, 1.0f,
                    0.0f, 1.0f
                };
                
                this->indices = {
                    // Indices
                    2, 1, 0,
                    0, 3, 2
                };
                this->load();
            }
            
            Element(const char *textureName) : Element() {
                this->texture = new Texture(textureName);
            };
            
            std::vector<float> get_vertices() {
                return vertices;
            }
            Texture *get_texture() {
                return texture;
            }
            
            virtual void handle_input(SDL_Event event) {}
          
            void load() { 
                // Adding the buffers' data
                glGenBuffers(1, &this->verticeVBO);
                glBindBuffer(GL_ARRAY_BUFFER, verticeVBO);
                glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
                
                
                glGenBuffers(1, &this->textureCoordVBO);
                glBindBuffer(GL_ARRAY_BUFFER, textureCoordVBO);
                glBufferData(GL_ARRAY_BUFFER, textureCoords.size() * sizeof(float), textureCoords.data(), GL_STATIC_DRAW);
                 
                glGenBuffers(1, &this->IBO);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);
                     
                glBindBuffer(GL_ARRAY_BUFFER, 0);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);  
            }
            
            virtual void render() {
                 // Do this after using the shader
                 if (this->texture != nullptr) {
                     this->texture->draw();
                     
                     GLint position = Shaders::get().find("uiShader")->attribute_location("a_pos");
                     GLint color = Shaders::get().find("uiShader")->attribute_location("a_colors");
                     GLint textureCoord = Shaders::get().find("uiShader")->attribute_location("a_tCoords");
                     
                     Mat4x4 move, scale;
                     move.set_translation(this->position.x, this->position.y, 0.0f);
                     scale.set_scaling(scaling.x, scaling.y, 0.0f);
                     Shaders::get().find("uiShader")->set_uniform_mat4("u_model", move.multiply(scale));
                     
                     
                     // Upload attribute data
                     glBindBuffer(GL_ARRAY_BUFFER, verticeVBO);
                     glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, 0);
                     glEnableVertexAttribArray(position);
                     
                     
                     glBindBuffer(GL_ARRAY_BUFFER, textureCoordVBO);
                     glVertexAttribPointer(textureCoord, 2, GL_FLOAT, GL_FALSE, 0, 0);
                     glEnableVertexAttribArray(textureCoord);
                     
                     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IBO);
                     glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
                     
                     glDisableVertexAttribArray(position);
                     glDisableVertexAttribArray(textureCoord);
                     glBindTexture(GL_TEXTURE_2D, 0);
                 }
            }
            Element *select(std::function<bool()> selection) {
                 this->selected = selection;
                 
                 return this;
            }
            void dispose() {
                 glDeleteBuffers(1, &this->verticeVBO);
                 glDeleteBuffers(1, &this->textureCoordVBO);
                 glDeleteBuffers(1, &this->IBO);
            }
            
        protected:
            std::vector<float> vertices;
            std::vector<float> textureCoords;
            std::vector<GLuint> indices;
            std::function<bool()> selected; 
            
            Texture *texture;
            
            GLuint verticeVBO;
            GLuint textureCoordVBO;
            GLuint IBO;
    };
    class ImageButton : public Element {
        public:
           ImageButton() : Element() {}
           
           ImageButton(const char *imageName) : Element(imageName) {}
           
           ImageButton(const char *imageName, std::function<void()> clickListener) : Element(imageName) {
               this->clickListener = clickListener;
           }
           
           void handle_input(SDL_Event event) override {
               int w = 0, h = 0;
               SDL_GetWindowSize(windows, &w, &h);
               int dx, dy;
               SDL_GetMouseState(&dx, &dy);
               float cx = dx, cy = dy;
               
               // Normalized coordinates
               cx /= w;
               cy /= h;
               
               cx *= SCREEN_WIDTH * 2;
               cy *= SCREEN_HEIGHT * 2;
               cx -= SCREEN_WIDTH;
               cy -= SCREEN_HEIGHT;
               cy *= -1;
               
               cx /= SCREEN_WIDTH;
               cy /= SCREEN_HEIGHT;
               
               bool intersecting = (cx >= position.x - scaling.x && cy >= position.y - scaling.y) && 
                                   (cx < position.x + scaling.x && cy < position.y + scaling.y);
               
               if (intersecting) { 
                   if (event.type == SDL_MOUSEMOTION) {
                       highlighted = true;
                       focused = true;
                   }
                  
                   if (event.type == SDL_MOUSEBUTTONUP) {
                       clickListener();
                       highlighted = false;
                       focused = false;
                   }
                } else {
                   highlighted = false;
                   focused = false;
                }
           }
        protected:
           bool highlighted;
           std::function<void()> clickListener;
    };
    class AtlasButton : public ImageButton {
        protected:
           BlockTypes selectType;
        public:
           // Horizontal atlas entry
           int atlasEntry;
           AtlasButton() : ImageButton() {}
           AtlasButton(const char *imageName, int entry, BlockTypes to) {
               this->texture = new Texture(imageName);
               this->atlasEntry = entry;
               this->selectType = to;
               
               int atlasEntries = 16;
               float tileSize = 16.0f;
               float width = atlasEntries * tileSize;
               float entryTile = entry * tileSize;
               
               // Calculate atlas texture coordinates
               // Top-left coordinates
               float ax1 = float(entryTile / width);
               float ay1 = 0.0f;
               
               // Bottom-right coordinates
               float ax2 = float((entryTile + tileSize) / width);
               float ay2 = 1.0f;
               
               this->vertices = {
                   // Vertices
                   -0.5f, -0.5f, 0.0f,
                   0.5f, -0.5f, 0.0f,
                   0.5f, 0.5f, 0.0f,
                   -0.5f, 0.5f, 0.0f
               };
                
               this->textureCoords = {
                    ax1, -ay2,
                    ax2, -ay2,
                    ax2, -ay1,
                    ax1, -ay1
               };
                
               this->indices = {
                    // Indices
                    2, 1, 0,
                    0, 3, 2
               };
               this->load();
           }
           void handle_input(SDL_Event event) override {
               // I know this is repetitive, but I've encountered problems with
               // building block type assignment when I used lambda functions
               int w = 0, h = 0;
               SDL_GetWindowSize(windows, &w, &h);
               int dx, dy;
               SDL_GetMouseState(&dx, &dy);
               float cx = dx, cy = dy;
               
               // Normalized coordinates
               cx /= w;
               cy /= h;
               
               cx *= SCREEN_WIDTH * 2;
               cy *= SCREEN_HEIGHT * 2;
               cx -= SCREEN_WIDTH;
               cy -= SCREEN_HEIGHT;
               cy *= -1;
               
               cx /= SCREEN_WIDTH;
               cy /= SCREEN_HEIGHT;
               
               bool intersecting = (cx >= position.x - scaling.x && cy >= position.y - scaling.y) && 
                                   (cx < position.x + scaling.x && cy < position.y + scaling.y);
               
               if (intersecting) { 
                   if (event.type == SDL_MOUSEMOTION) {
                       highlighted = true;
                       focused = true;
                   }
                  
                   if (event.type == SDL_MOUSEBUTTONUP) {
                       player->set_selected(selectType);
                       highlighted = false;
                       focused = false;
                   }
                } else {
                   highlighted = false;
                   
                }
           }
    };
    
    std::vector<Element*> elements;
    
    void handle_event(SDL_Event event) {
        for (auto &element : elements) {
             element->handle_input(event);
        }
    }
    void render() {
        Shaders::get().use("uiShader");
        
        for (auto &element : elements) {
             element->render();
        }
    }
    
    
    
    
    void add(Element *element, float x, float y, float scalingX, float scalingY) {
        element->position.x = x;
        element->position.y = y;
        element->scaling.x = scalingX;
        element->scaling.y = scalingY;
        
        elements.push_back(element);
    }
    
    void add(Element *element, float x, float y) {
        add(element, x, y, 1.0f, 1.0f);
    }
    
    void add_item(BlockTypes type, int columns, int size) {
        auto s = [&]() -> bool {
             return (player->selectedBlock == type);
        };
        AtlasButton *a = new AtlasButton("textures.png", get_index_from_type(type), type);
        a->select(s);
        
        add(a, size * -1.0 / 20.0f + columns / 10.0f, -0.9f, 0.1f, 0.1f);
    }
    
    void load() {
       // Add crosshair
       add(new Element("crosshair.png"), 0.0, 0.0, 0.03f, 0.05f);
       
       int i = 0;
       std::vector<BlockTypes> types = {
           BlockTypes::dirt, BlockTypes::woodLog,
           BlockTypes::woodenPlanks, BlockTypes::glass,
           BlockTypes::window, BlockTypes::bricks,
           BlockTypes::stone, BlockTypes::cobblestone
       };
       
       for (auto &m : types) {
            add_item(m, i, types.size());
            i++;
       }
       
       for (auto &element : elements) {
           if (element->get_texture() != nullptr) {
               element->get_texture()->load();
           }
       }
    }
    
    void dispose() {
        for (auto &obj : elements) {
             obj->dispose();
        }
    }
};

class Game
{
  public:
    const char *displayName = "";
    virtual ~Game() {};
    virtual void init() {};
    virtual void load() {};
    
    virtual void handle_event(SDL_Event ev, float timeTook) {};

    virtual void update(float timeTook) {};
    virtual void dispose() {};
};

class GLCraft : public Game { 
    float time;
    public:  
       void init() override {
           displayName = "GL Craft";
       }
       void load() override {
           Perlin::load_perm_table();
           Shaders::get().load();
           Projection::reset();
           
           time = 0.0f;
           
           level = new ChunkCollection();
           player = new Player();
           UI::load();
           
           
       }
       void handle_event(SDL_Event ev, float timeTook) override {
           UI::handle_event(ev);
           if (!UI::focused) {
               int w = 0, h = 0;
               SDL_GetWindowSize(windows, &w, &h);
               Vec2i click = this->get_mouse_position(ev);
           
               if (ev.type == SDL_MOUSEMOTION) {       
                   float sensitivity = 0.1f;
               
                   if (click.y < h / 2) {
                       player->rotationX -= ev.motion.xrel * sensitivity * timeTook;
                       player->rotationY -= ev.motion.yrel * sensitivity * timeTook;
                   }
               
                   if (player->rotationX > 2 * M_PI) player->rotationX = 0;
                   if (player->rotationX < 0) player->rotationX = 2 * M_PI;
               
                   if (player->rotationY > (89.0f / 180.0f * M_PI)) player->rotationY = (89.0f / 180.0f * M_PI);
                   if (player->rotationY < -(89.0f / 180.0f * M_PI)) player->rotationY = -(89.0f / 180.0f * M_PI);
           
               
               
                   float s = player->speed;
                   if (click.y > h / 2 && click.x < w / 2) {
                       player->vel.x += cos(player->rotationX) * s * timeTook;
                       player->vel.z += sin(player->rotationX) * s * timeTook;
                   }
                   if (click.y > h / 2 && click.x > w / 2) {
                       player->vel.x -= cos(player->rotationX) * s * timeTook;
                       player->vel.z -= sin(player->rotationX) * s * timeTook;
                   }
                   if (click.y > h - 100.0) {
                       player->jump();
                   }
               }
               
           
               if (ev.type == SDL_MOUSEBUTTONUP) {
                   if (click.y < 100) {
                       if (click.x < w / 2) {
                           if (player->selectedBlock != BlockTypes::air) {   
                               Vec3i dda = player->ray->update_DDA(true);
                               level->set_block(dda.x, dda.y, dda.z, player->selectedBlock);
                           }    
                       } else {
                           Vec3i dda = player->ray->update_DDA(false);
                           level->set_block(dda.x, dda.y, dda.z, BlockTypes::air);
                       }
                   }
               } 
           }
           UI::focused = false;
       }
       void update(float timeTook) override {
           time += timeTook;

           player->update(timeTook);
           
           Projection::lookingAt.x = Projection::cameraPosition.x + cos(player->rotationX) * cos(player->rotationY);
           Projection::lookingAt.y = Projection::cameraPosition.y + sin(player->rotationY);
           Projection::lookingAt.z = Projection::cameraPosition.z + sin(player->rotationX) * cos(player->rotationY);
           
           Projection::update();
           
           Shaders::get().use("testShader");
        
           //Shaders::get().find("testShader")->set_uniform_vec3f("lightPosition", player->position.x, player->position.y, player->position.z);
           Shaders::get().find("testShader")->set_uniform_vec3f("lightPosition", 0.0, 50.0, 0.0);
           
           glEnable(GL_DEPTH_TEST);
           glEnable(GL_POLYGON_OFFSET_FILL);
           level->render();
           
           glDisable(GL_POLYGON_OFFSET_FILL);
           player->render_outline(time);
           
           glDisable(GL_DEPTH_TEST);
           
           UI::render();
       }
       Vec2i get_mouse_position(SDL_Event event) {
           int dx = 0, dy = 0;
           SDL_GetMouseState(&dx, &dy);
           Vec2i result = { dx, dy };
           
           return result;
       }
       
       void dispose() override {
           Shaders::get().clear();
           level->dispose();
           UI::dispose();
           player->cleanup();
       }
};

int main()
{
	if (SDL_Init(SDL_INIT_EVERYTHING) != 0)
	{
		fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
		return 1;
	}
    
	// We use OpenGL ES 2.0
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

	// We want at least 8 bits per color
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    
    // 1 bit per alpha
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 1);
    
    
    GLCraft game;
    game.init();
    
	SDL_Window *window = SDL_CreateWindow("GL Craft", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_OPENGL);
	if (window == NULL)
	{
		fprintf(stderr, "SDL_CreateWindow Error: %s\n", SDL_GetError());
		return 1;
	}
	windows = window;
	
	// We will not actually need a context created, but we should create one
	SDL_GLContext context = SDL_GL_CreateContext(window);
    
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    
    // Constant OpenGL settings:
    glClearColor(0.4f, 0.5f, 0.9f, 1.0f);
    	
	glDepthFunc(GL_LESS);
	glCullFace(GL_FRONT);
    glFrontFace(GL_CCW);
    glPolygonOffset(1, 1);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	
    game.load();
    
    
	float then = 0.0f, delta = 0.0f;
    bool disabled = false;
    SDL_Event e;
    while (!disabled)
	{
		now = SDL_GetTicks();
        delta = (now - then) * 1000 / SDL_GetPerformanceFrequency();
        then = now;
		while (SDL_PollEvent(&e))
		{
			switch (e.type)
            {
                 case SDL_QUIT:
                      disabled = true;
                      break;
            }
            // Event-handling code
            game.handle_event(e, delta);
		}
		
		// Drawing
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    	
    	// Update and render to screen code
    	game.update(delta);
    	
		// Swap buffers
		SDL_GL_SwapWindow(window);
	}
	game.dispose();
	
    SDL_GL_DeleteContext(context);
    
	SDL_DestroyWindow(window);
	SDL_Quit();
	IMG_Quit();
	
	return 0;
}