/***************************************************************************
 *   Copyright (c) 2024 Derek Sauer <dereksauer.ca@gmail.com>              *
 *                                                                         *
 *   This file is part of the FreeCAD CAx development system.              *
 *                                                                         *
 *   This library is free software; you can redistribute it and/or         *
 *   modify it under the terms of the GNU Library General Public           *
 *   License as published by the Free Software Foundation; either          *
 *   version 2 of the License, or (at your option) any later version.      *
 *                                                                         *
 *   This library  is distributed in the hope that it will be useful,      *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU Library General Public License for more details.                  *
 *                                                                         *
 *   You should have received a copy of the GNU Library General Public     *
 *   License along with this library; see the file COPYING.LIB. If not,    *
 *   write to the Free Software Foundation, Inc., 59 Temple Place,         *
 *   Suite 330, Boston, MA  02111-1307, USA                                *
 *                                                                         *
 ***************************************************************************/

#include "View3DInventorViewer.h"
#include "View3DGradientBackground.h"

// TODO: Determine if every Inventor Viewer shares the same OpenGL context. If so, we can use a
// simple reference counted resource wrapper around the GL objects we create so we only pay
// for the compilation time and GPU memory cost once. The ICD is almost certainly caching the shader
// programs so compilation time and memory cost on new viewer windows will be nearly nil but still
// poor practice to go through the motions needlessly.

namespace Gui
{
// The vertex shader that transforms the full screen quad binds its vertex
// coordinates to attribute slot 0, and its UV coordinates to attribute slot 1.
const GLuint View3DGradientBackground::vertex_coord_attribute_slot = 0;
const GLuint View3DGradientBackground::uv_coord_attribute_slot = 1;

// All gradient rendering fragment shaders use the same uniform names to pass
// gradient parameters to the shader.
const std::string View3DGradientBackground::from_color_uniform = "from_color";
const std::string View3DGradientBackground::to_color_uniform = "to_color";
const std::string View3DGradientBackground::mid_color_uniform = "mid_color";
const std::string View3DGradientBackground::angle_uniform = "angle";

/// Construct a new smooth gradient background renderer.
///
/// @param[in] viewer Pointer to the Inventor viewer that will use this gradient background renderer
/// to draw its background.
View3DGradientBackground::View3DGradientBackground(View3DInventorViewer* viewer)
    : viewer(viewer)
{}

/// Destruct this gradient background renderer.
View3DGradientBackground::~View3DGradientBackground()
{
    auto gl_functions = this->gl_functions();

    // If gl_functions() returns `nullopt` the context is already gone and
    // cleanup no longer matters.
    if (gl_functions.has_value()) {
        auto gl = gl_functions.value();

        // glDelete* functions silently ignore uninitialized or invalid handles.
        gl->glDeleteProgram(this->radial_3color_gradient_program);
        gl->glDeleteProgram(this->radial_2color_gradient_program);
        gl->glDeleteProgram(this->linear_3color_gradient_program);
        gl->glDeleteProgram(this->linear_2color_gradient_program);
        gl->glDeleteBuffers(1, &this->quad_vertex_buffer);
    }

    this->viewer = nullptr;
}

/// Prepare to render a gradient background.
///
/// When the gradient background renderer is first constructed, it has no ability to render a
/// background as the viewer itself has not finished initializing and has no valid OpenGL
/// context. The only time we can be sure the viewer's OpenGL context exists and is current is just
/// before the viewer renders a frame. Prepare our OpenGL resources "just in time".
///
/// @param[in] gl Pointer to OpenGL2.x functions exposed by the viewer's OpenGL context.
void View3DGradientBackground::prepare(QOpenGLFunctions* gl)
{
    // Failure to prepare the shader pipeline is not a show stopper. We can revert to the fixed
    // function pipeline, it just won't look as good. Regardless of this function's success, we are
    // still prepared to draw something.
    this->prepared_to_draw = true;

    // === Shader compilation stage ===========================================

    GLuint vs_shader_handle = this->compile_shader(gl,
                                                   ShaderType::VertexShader,
                                                   {View3DGradientBackground::vs_shader_src});
    if (vs_shader_handle == 0) {
        return;
    }

    GLuint fs_linear_2color_shader_handle =
        this->compile_shader(gl,
                             ShaderType::FragmentShader,
                             {View3DGradientBackground::fs_common_src,
                              View3DGradientBackground::fs_linear_2color_shader_src});
    if (fs_linear_2color_shader_handle == 0) {
        return;
    }

    GLuint fs_linear_3color_shader_handle =
        this->compile_shader(gl,
                             ShaderType::FragmentShader,
                             {View3DGradientBackground::fs_common_src,
                              View3DGradientBackground::fs_linear_3color_shader_src});
    if (fs_linear_3color_shader_handle == 0) {
        return;
    }

    GLuint fs_radial_2color_shader_handle =
        this->compile_shader(gl,
                             ShaderType::FragmentShader,
                             {View3DGradientBackground::fs_common_src,
                              View3DGradientBackground::fs_radial_2color_shader_src});
    if (fs_radial_2color_shader_handle == 0) {
        return;
    }

    GLuint fs_radial_3color_shader_handle =
        this->compile_shader(gl,
                             ShaderType::FragmentShader,
                             {View3DGradientBackground::fs_common_src,
                              View3DGradientBackground::fs_radial_3color_shader_src});
    if (fs_radial_3color_shader_handle == 0) {
        return;
    }

    // === Shader program linking stage =======================================

    this->linear_2color_gradient_program =
        this->link_shader_program(gl, {vs_shader_handle, fs_linear_2color_shader_handle});
    if (this->linear_2color_gradient_program == 0) {
        return;
    }

    this->linear_3color_gradient_program =
        this->link_shader_program(gl, {vs_shader_handle, fs_linear_3color_shader_handle});
    if (this->linear_3color_gradient_program == 0) {
        return;
    }

    this->radial_2color_gradient_program =
        this->link_shader_program(gl, {vs_shader_handle, fs_radial_2color_shader_handle});
    if (this->radial_2color_gradient_program == 0) {
        return;
    }

    this->radial_3color_gradient_program =
        this->link_shader_program(gl, {vs_shader_handle, fs_radial_3color_shader_handle});
    if (this->radial_3color_gradient_program == 0) {
        return;
    }

    // === Shader cleanup stage ===============================================

    // Shader source objects are no longer needed once the shader programs are linked.
    // glDeleteShader() silently ignores uninitialized or invalid handles.
    gl->glDeleteShader(fs_radial_3color_shader_handle);
    gl->glDeleteShader(fs_radial_2color_shader_handle);
    gl->glDeleteShader(fs_linear_3color_shader_handle);
    gl->glDeleteShader(fs_linear_2color_shader_handle);
    gl->glDeleteShader(vs_shader_handle);

    // === Vertex buffer stage ================================================

    // Vertex buffer format: x, y, u, v
    // clang-format off
    const std::array<GLfloat, 16> quad_vb_data = {
        -1.0F, -1.0F, 0.0F, 0.0F,
         1.0F, -1.0F, 1.0F, 0.0F,
        -1.0F,  1.0F, 0.0F, 1.0F,
         1.0F,  1.0F, 1.0F, 1.0F
    };
    // clang-format on

    GLsizei quad_vb_data_size = quad_vb_data.size() * sizeof(GLfloat);

    gl->glGenBuffers(1, &this->quad_vertex_buffer);
    if (this->quad_vertex_buffer == 0) {
        Base::Console().Warning(
            "View3DGradientBackground - Vertex buffer handle could not be created.");
        return;
    }

    gl->glBindBuffer(GL_ARRAY_BUFFER, this->quad_vertex_buffer);
    gl->glBufferData(GL_ARRAY_BUFFER, quad_vb_data_size, quad_vb_data.data(), GL_STATIC_DRAW);

    GLint vb_buffer_size = 0;
    gl->glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &vb_buffer_size);
    if (vb_buffer_size != quad_vb_data_size) {
        Base::Console().Warning("View3DGradientBackground - Vertex buffer upload failed.");
        return;
    }

    gl->glBindBuffer(GL_ARRAY_BUFFER, 0);


    this->gl_init_success = true;
}

/// Draw a linear gradient as the viewer's background image.
void View3DGradientBackground::draw_linear_gradient()
{
    auto gl_functions = this->gl_functions();

    if (gl_functions.has_value()) {
        auto gl = gl_functions.value();

        if (!this->prepared_to_draw) {
            this->prepare(gl);
        }

        if (this->gl_init_success) {
            GLuint program_handle = this->mid_color.has_value()
                ? this->linear_3color_gradient_program
                : this->linear_2color_gradient_program;

            gl->glUseProgram(program_handle);

            gl->glBindBuffer(GL_ARRAY_BUFFER, this->quad_vertex_buffer);

            gl->glEnableVertexAttribArray(View3DGradientBackground::vertex_coord_attribute_slot);
            gl->glVertexAttribPointer(View3DGradientBackground::vertex_coord_attribute_slot,
                                      2,
                                      GL_FLOAT,
                                      GL_FALSE,
                                      sizeof(GLfloat) * 4,
                                      0);

            gl->glEnableVertexAttribArray(View3DGradientBackground::uv_coord_attribute_slot);
            gl->glVertexAttribPointer(View3DGradientBackground::uv_coord_attribute_slot,
                                      2,
                                      GL_FLOAT,
                                      GL_FALSE,
                                      sizeof(GLfloat) * 4,
                                      (void*)(sizeof(GLfloat) * 2));

            // TODO: No need to do this every frame. Create a shader program wrapper class that
            // caches uniform locations and only updates uniforms when the background parameters
            // change.
            auto from_color_uniform_loc =
                gl->glGetUniformLocation(program_handle,
                                         View3DGradientBackground::from_color_uniform.c_str());
            gl->glUniform3f(from_color_uniform_loc,
                            this->from_color.redF(),
                            this->from_color.greenF(),
                            this->from_color.blueF());

            if (this->mid_color.has_value()) {
                auto mid_color_uniform_loc =
                    gl->glGetUniformLocation(program_handle,
                                             View3DGradientBackground::mid_color_uniform.c_str());
                gl->glUniform3f(mid_color_uniform_loc,
                                this->mid_color->redF(),
                                this->mid_color->greenF(),
                                this->mid_color->blueF());
            }

            auto to_color_uniform_loc =
                gl->glGetUniformLocation(program_handle,
                                         View3DGradientBackground::to_color_uniform.c_str());
            gl->glUniform3f(to_color_uniform_loc,
                            this->to_color.redF(),
                            this->to_color.greenF(),
                            this->to_color.blueF());

            auto angle_uniform_loc =
                gl->glGetUniformLocation(program_handle,
                                         View3DGradientBackground::angle_uniform.c_str());
            gl->glUniform1f(angle_uniform_loc, this->angle);

            gl->glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

            gl->glDisableVertexAttribArray(View3DGradientBackground::uv_coord_attribute_slot);
            gl->glDisableVertexAttribArray(View3DGradientBackground::vertex_coord_attribute_slot);
            gl->glBindBuffer(GL_ARRAY_BUFFER, 0);
            gl->glUseProgram(0);
        }
        else {
            // TODO: Reimplement the fixed function pipeline gradient drawing routine as fallback.
            // TODO: Discuss omitting this fallback entirely with FreeCAD's maintainers.
            //       FreeCAD targets OpenGL 2.0 compatibility profile and this class only uses
            //       functionality supported by OpenGL 2.0 and OpenGL ES 2.0 with no extensions
            //       needed.
            gl->glClearColor(this->from_color.redF(),
                             this->from_color.greenF(),
                             this->from_color.blueF(),
                             1.0F);
            gl->glClear(GL_COLOR_BUFFER_BIT);
        }
    }
}

/// Draw a radial gradient as the viewer's background image.
void View3DGradientBackground::draw_radial_gradient()
{
    auto gl_functions = this->gl_functions();

    if (gl_functions.has_value()) {
        auto gl = gl_functions.value();

        if (!this->prepared_to_draw) {
            this->prepare(gl);
        }

        if (this->gl_init_success) {
            GLuint program_handle = this->mid_color.has_value()
                ? this->radial_3color_gradient_program
                : this->radial_2color_gradient_program;

            gl->glUseProgram(program_handle);

            gl->glBindBuffer(GL_ARRAY_BUFFER, this->quad_vertex_buffer);

            gl->glEnableVertexAttribArray(View3DGradientBackground::vertex_coord_attribute_slot);
            gl->glVertexAttribPointer(View3DGradientBackground::vertex_coord_attribute_slot,
                                      2,
                                      GL_FLOAT,
                                      GL_FALSE,
                                      sizeof(GLfloat) * 4,
                                      0);

            gl->glEnableVertexAttribArray(View3DGradientBackground::uv_coord_attribute_slot);
            gl->glVertexAttribPointer(View3DGradientBackground::uv_coord_attribute_slot,
                                      2,
                                      GL_FLOAT,
                                      GL_FALSE,
                                      sizeof(GLfloat) * 4,
                                      (void*)(sizeof(GLfloat) * 2));

            // TODO: No need to do this every frame. Create a shader program wrapper class that
            // caches uniform locations and only updates uniforms when the background parameters
            // change.
            auto from_color_uniform_loc =
                gl->glGetUniformLocation(program_handle,
                                         View3DGradientBackground::from_color_uniform.c_str());
            gl->glUniform3f(from_color_uniform_loc,
                            this->from_color.redF(),
                            this->from_color.greenF(),
                            this->from_color.blueF());

            if (this->mid_color.has_value()) {
                auto mid_color_uniform_loc =
                    gl->glGetUniformLocation(program_handle,
                                             View3DGradientBackground::mid_color_uniform.c_str());
                gl->glUniform3f(mid_color_uniform_loc,
                                this->mid_color->redF(),
                                this->mid_color->greenF(),
                                this->mid_color->blueF());
            }

            auto to_color_uniform_loc =
                gl->glGetUniformLocation(program_handle,
                                         View3DGradientBackground::to_color_uniform.c_str());
            gl->glUniform3f(to_color_uniform_loc,
                            this->to_color.redF(),
                            this->to_color.greenF(),
                            this->to_color.blueF());

            gl->glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

            gl->glDisableVertexAttribArray(View3DGradientBackground::uv_coord_attribute_slot);
            gl->glDisableVertexAttribArray(View3DGradientBackground::vertex_coord_attribute_slot);
            gl->glBindBuffer(GL_ARRAY_BUFFER, 0);
            gl->glUseProgram(0);
        }
        else {
            // TODO: Reimplement the fixed function pipeline gradient drawing routine as fallback.
            // TODO: Discuss omitting this fallback entirely with FreeCAD's maintainers.
            //       FreeCAD targets OpenGL 2.0 compatibility profile and this class only uses
            //       functionality supported by OpenGL 2.0 and OpenGL ES 2.0 with no extensions
            //       needed.
            gl->glClearColor(this->from_color.redF(),
                             this->from_color.greenF(),
                             this->from_color.blueF(),
                             1.0F);
            gl->glClear(GL_COLOR_BUFFER_BIT);
        }
    }
}

/// Set the colors of the background gradient.
///
/// @param[in] from_color Starting color.
/// @param[in] to_color Ending color.
/// @param[in] mid_color Optional middle color.
void View3DGradientBackground::set_gradient_colors(const QColor& from_color,
                                                   const QColor& to_color,
                                                   const std::optional<QColor>& mid_color)
{
    this->from_color = from_color;
    this->to_color = to_color;
    this->mid_color = mid_color;
}

/// Retrieve the OpenGL 2.x functions exposed by viewer's OpenGL context.
///
/// @return Returns an optional that contains either a pointer to the OpenGL functions exposed by
/// the viewer's OpenGL context; or nothing.
std::optional<QOpenGLFunctions*> View3DGradientBackground::gl_functions()
{
    if (this->viewer) {
        auto gl_widget = qobject_cast<QtGLWidget*>(this->viewer->getGLWidget());

        if (gl_widget) {
            auto gl_context = gl_widget->context();

            if (gl_context) {
                return gl_context->functions();
            }
        }
    }

    return std::nullopt;
}

/// Compile a shader object ready to be linked into a shader program.
///
/// @param[in] gl Pointer to OpenGL2.x functions exposed by the viewer's OpenGL context.
/// @param[in] shader_type Indicates the creation of a vertex or fragment shader.
/// @param[in] shader_source A collection of strings containing the source code that comprises this
/// shader. Multiple strings will be concatated prior to compiling.
///
/// @return On success, returns the OpenGL handle to the newly compiled shader. On failure, returns
/// zero, and logs compiler errors to the console as warnings.
///
/// @remarks Error messages are logged as warnings since failure to a shader is not a
/// show stopper, we will revert to rendering the gradient with the fixed function pipeline.
GLuint View3DGradientBackground::compile_shader(QOpenGLFunctions* gl,
                                                View3DGradientBackground::ShaderType shader_type,
                                                const std::vector<std::string>& shader_source)
{
    GLuint shader_handle = gl->glCreateShader(shader_type);
    if (shader_handle == 0) {
        Base::Console().Warning(
            "View3DGradientBackground - Could not create a new OpenGL shader handle.\n");
        return 0;
    }

    // glShaderSource expects a pointer to an array of c-strings.
    // Remap our collection of C++ strings into a format the function can understand.
    std::vector<const char*> cstr_collection;
    std::transform(shader_source.begin(),
                   shader_source.end(),
                   back_inserter(cstr_collection),
                   mem_fn(&std::string::c_str));
    const char** shader_source_cstr_array = cstr_collection.data();

    gl->glShaderSource(shader_handle, shader_source.size(), shader_source_cstr_array, nullptr);

    gl->glCompileShader(shader_handle);

    GLint compile_success = 0;
    gl->glGetShaderiv(shader_handle, GL_COMPILE_STATUS, &compile_success);

    if (compile_success == GL_FALSE) {
        GLint compile_msg_length = 0;
        gl->glGetShaderiv(shader_handle, GL_INFO_LOG_LENGTH, &compile_msg_length);

        std::string compile_msg;
        compile_msg.reserve(compile_msg_length);
        gl->glGetShaderInfoLog(shader_handle, compile_msg_length, nullptr, compile_msg.data());

        Base::Console().Warning(
            "View3DGradientBackground - Failed to compile shader with error(s):\n%s\n",
            compile_msg.c_str());

        gl->glDeleteShader(shader_handle);

        return 0;
    }

    return shader_handle;
}

/// Link one or more shaders into a shader program.
///
/// @param[in] gl Pointer to OpenGL2.x functions exposed by the viewer's OpenGL context.
/// @param[in] compiled_shaders A collection of OpenGL shaders handles.
///
/// @return On success, returns the OpenGL handle to the newly linked shader program. On failure,
/// returns zero, and logs compiler errors to the console as warnings.
///
/// @remarks Error messages are logged as warnings since failure to link a shader program is not a
/// show stopper, we will revert to rendering the gradient with the fixed function pipeline.
GLuint View3DGradientBackground::link_shader_program(QOpenGLFunctions* gl,
                                                     const std::vector<GLuint>& compiled_shaders)
{
    GLuint program_handle = gl->glCreateProgram();
    if (program_handle == 0) {
        Base::Console().Warning(
            "View3DGradientBackground - Unable to create a new shader program handle.");
        return 0;
    }

    for (auto shader_handle : compiled_shaders) {
        gl->glAttachShader(program_handle, shader_handle);
    }

    gl->glBindAttribLocation(program_handle,
                             View3DGradientBackground::vertex_coord_attribute_slot,
                             "vertex_coord");
    gl->glBindAttribLocation(program_handle,
                             View3DGradientBackground::uv_coord_attribute_slot,
                             "uv_coord");

    gl->glLinkProgram(program_handle);

    GLint link_success = 0;
    gl->glGetProgramiv(program_handle, GL_LINK_STATUS, &link_success);

    if (link_success == GL_FALSE) {
        GLint link_msg_length = 0;
        gl->glGetProgramiv(program_handle, GL_INFO_LOG_LENGTH, &link_msg_length);

        std::string link_msg;
        link_msg.reserve(link_msg_length);
        gl->glGetProgramInfoLog(program_handle, link_msg_length, nullptr, link_msg.data());

        Base::Console().Warning(
            "View3DGradientBackground - Failed to link shader program with error(s):\n%s\n",
            link_msg.c_str());

        gl->glDeleteProgram(program_handle);

        return 0;
    }

    // Once a shader program is linked, the compiled shader code comprising the program is no longer
    // needed. If the shader handles are still attached to a program, the GL runtime merely flags
    // the shader handles for later deletion when the shader program they are attached to is
    // destroyed. Detaching allows the runtime to destroy the shader handles as soon as possible,
    // liberating that memory.
    for (auto shader_handle : compiled_shaders) {
        gl->glDetachShader(program_handle, shader_handle);
    }

    return program_handle;
}

/// Vertex shader that performs no transformations on incoming vertices
/// and passes UV coordinates unchanged to the fragment shader.
///
/// @param[in] vertex_coord Two element (X & Y) vertex position bound to attribute slot 0.
/// @param[in] uv_coord Vertex UV coordinate bound to attribute slot 1.
const std::string View3DGradientBackground::vs_shader_src = R"(
    #version 110

    #ifdef GL_ES
        precision mediump float;
    #endif

    attribute vec2 vertex_coord;
    attribute vec2 uv_coord;

    varying vec2 uv;

    void main() {
        uv = uv_coord;
        gl_Position = vec4(vertex_coord, 0.0, 1.0);
    }
)";

/// Code common to all gradient fragment shaders.
const std::string View3DGradientBackground::fs_common_src = R"(
    #version 110

    #ifdef GL_ES
        precision mediump float;
    #endif

    /// Pseudo random gradient noise generation function.
    /// 
    /// This function was presented by Jorge Jimenze during his talk at SIGGRAPH 2014 and as part of
    /// the material for SIGGRAPH's "Advances in Real-time Rendering" course.
    ///
    /// Course and presentation available here:
    /// https://advances.realtimerendering.com/s2014/index.html#_NEXT_GENERATION_POST
    ///
    /// @param[in] fragment_coord Window space coordinate of the fragment (gl_FragCoord.xy).
    ///
    /// @remarks Window space coordinates are used to ensure the noise remains stable across the
    ///          viewport to avoid shimmering.
    float gradient_noise(in vec2 fragment_coord)
    {
    	  return fract(52.9829189 * fract(dot(fragment_coord, vec2(0.06711056, 0.00583715))));
    }

    /// Apply a dithering function to a fragment's color.
    ///
    /// Pseudo random noise will be mixed into the fragment color. A single 8-bit grayscale
    /// value worth of noise will be added to the original color then backed off by the average
    /// brightness of an 8-bit grayscale value. This nearly imperceptible change to the color's value
    /// will break up apparent banding in color gradients.
    ///
    /// @param[in] fragment_color RGB color of the fragment.
    /// @param[in] fragment_coord Window space coordinate of the fragment (gl_FragCoord.xy).
    ///
    /// @returns The original fragment color subtly mixed with noise.
    vec3 dither(vec3 fragment_color, vec2 fragment_coord) {
        // A single eight bit grayscale value. 
        const float grayscale_value = 1.0 / 255.0;

        // Average brightness of a single grayscale value.
        const float average_brightness = 0.5 / 255.0;

        return fragment_color += grayscale_value * gradient_noise(fragment_coord) - average_brightness;
    }
   

    /// Rotate a UV coordinate.
    ///
    /// @param[in] uv The UV coordinate to rotate.
    /// @param[in] angle Angle to rotate in degrees.
    vec2 rotate_uv(vec2 uv, float angle) {
        angle = radians(angle);

        return vec2(
            cos(angle) * uv.x + sin(angle) * uv.y,
            cos(angle) * uv.y + sin(angle) * uv.x
        );
    }

    /// OpenGL ES compatible function to convert a single color channel from sRGB to linear color space.
    /// Reference: https://entropymine.com/imageworsener/srgbformula/
    float srgb_to_linear_f(float color) {
        if (color <= 0.04045) {
            return color / 12.92;
        } else {
            return pow((color + 0.055) / 1.055, 2.4);
        }
    }

    /// Convert three color RGB from sRGB colorspace to linear.
    vec3 to_linear(vec3 color) {
        return vec3(srgb_to_linear_f(color.r),
                    srgb_to_linear_f(color.g),
                    srgb_to_linear_f(color.b));
    }

    /// OpenGL ES compatible function to convert a single color channel from linear to sRGB color space.
    /// Reference: https://entropymine.com/imageworsener/srgbformula/
    float to_srgb_f(float color) {
        if (color < 0.0031308) {
            return color * 12.92;
        } else {
            return 1.055 * pow(color, 1.0 / 2.4) - 0.055;
        }
    }

    /// Convert three color RGB from linear colorspace to sRGB.
    vec3 to_srgb(vec3 color) {
        return vec3(to_srgb_f(color.r),
                    to_srgb_f(color.g),
                    to_srgb_f(color.b));
    }

    /// Mix linear RGB colors in OKLAB color space.
    /// Reference: https://www.shadertoy.com/view/ttcyRS
    /// MIT License
    /// Copyright © 2020 Inigo Quilez
    vec3 oklab_mix(vec3 color_a, vec3 color_b, float h)
    {
        const mat3 kCONEtoLMS = mat3(                
             0.4121656120,  0.2118591070,  0.0883097947,
             0.5362752080,  0.6807189584,  0.2818474174,
             0.0514575653,  0.1074065790,  0.6302613616);

        const mat3 kLMStoCONE = mat3(
             4.0767245293, -1.2681437731, -0.0041119885,
            -3.3072168827,  2.6093323231, -0.7034763098,
             0.2307590544, -0.3411344290,  1.7068625689);
                    
        vec3 lmsA = pow(kCONEtoLMS * color_a, vec3(1.0 / 3.0));
        vec3 lmsB = pow(kCONEtoLMS * color_b, vec3(1.0 / 3.0));

        vec3 lms = mix(lmsA, lmsB, h);

        return kLMStoCONE * (lms * lms * lms);
    }
)";

// TODO: Collapse gradient functions into single function versions.
//       Linear gradient and radial gradient can both be accomplished
//       with a single function each that loops through the gradient stops.
//       Dedicated two and three color variants are not needed.

/// Fragment shader drawing a two color linear gradient with angle.
///
/// @param[in] uv Fragment's UV coordinates passed from the vertex shader.
/// @param[in] from_color Start color of the gradient.
/// @param[in] to_color End color of the gradient.
/// @param[in] angle Angle of the gradient, in degrees.
const std::string View3DGradientBackground::fs_linear_2color_shader_src = R"(
    varying vec2 uv;

    uniform vec3 from_color;
    uniform vec3 to_color;
    uniform float angle;

    void main() {
        // Offset towards center of UV and rotate.
        vec2 rotated_uv = uv - 0.5;
        rotated_uv = rotate_uv(rotated_uv, angle);
        rotated_uv += 0.5;

        // Convert incoming colors to linear color space
        vec3 from_color_linear = to_linear(from_color);
        vec3 to_color_linear = to_linear(to_color);

        // Use OKLAB color space to mix the gradient for silky smooth transitions.
        vec3 fragment_color = oklab_mix(from_color_linear, to_color_linear, rotated_uv.x);

        fragment_color = dither(fragment_color, gl_FragCoord.xy);

        gl_FragColor = vec4(to_srgb(fragment_color), 1.0);
    }
)";

/// Fragment shader drawing a three color linear gradient with angle.
///
/// @param[in] uv Fragment's UV coordinates passed from the vertex shader.
/// @param[in] from_color Start color of the gradient.
/// @param[in] to_color End color of the gradient.
/// @param[in] mid_color Middle color of the gradient.
/// @param[in] angle Angle of the gradient, in degrees.
const std::string View3DGradientBackground::fs_linear_3color_shader_src = R"(
    varying vec2 uv;

    uniform vec3 from_color;
    uniform vec3 to_color;
    uniform vec3 mid_color;
    uniform float angle;

    void main() {
        // Offset towards center of UV and rotate.
        vec2 rotated_uv = uv - 0.5;
        rotated_uv = rotate_uv(rotated_uv, angle);
        rotated_uv += 0.5;

        // Convert incoming colors to linear color space for gradient calculations.
        vec3 from_color_linear = to_linear(from_color);
        vec3 to_color_linear = to_linear(to_color);
        vec3 mid_color_linear = to_linear(mid_color);

        vec3 fragment_color = oklab_mix(from_color_linear, mid_color_linear, smoothstep(0.0, 0.5, rotated_uv.x));
        fragment_color = oklab_mix(fragment_color, to_color_linear, smoothstep(0.5, 1.0, rotated_uv.x));

        fragment_color = dither(fragment_color, gl_FragCoord.xy);

        gl_FragColor = vec4(to_srgb(fragment_color), 1.0);
    }
)";

/// Fragment shader drawing a two color radial gradient.
///
/// @param[in] uv Fragment's UV coordinates passed from the vertex shader.
/// @param[in] from_color Start color of the gradient.
/// @param[in] to_color End color of the gradient.
const std::string View3DGradientBackground::fs_radial_2color_shader_src = R"(
    varying vec2 uv;

    uniform vec3 from_color;
    uniform vec3 to_color;

    void main() {
        // Offset UV coordinates to center of viewport and scale them up slightly.
        // Without some scaling the `to_color` will be barely visible in the corners of the viewport.
        float scale_factor = 1.3;
        vec2 centered_uv = scale_factor * uv - (scale_factor * 0.5);
        float uv_length = length(centered_uv);

        // Convert incoming colors to linear color space for gradient calculations.
        vec3 from_color_linear = to_linear(from_color);
        vec3 to_color_linear = to_linear(to_color);

        // Generate the gradient in OKLAB colorspace.
        vec3 fragment_color = oklab_mix(from_color_linear, to_color_linear, uv_length);

        fragment_color = dither(fragment_color, gl_FragCoord.xy);

        gl_FragColor = vec4(to_srgb(fragment_color), 1.0);
    }
)";

/// Fragment shader drawing a three color radial gradient.
///
/// @param[in] uv Fragment's UV coordinates passed from the vertex shader.
/// @param[in] from_color Start color of the gradient.
/// @param[in] to_color End color of the gradient.
/// @param[in] mid_color Middle color of the gradient.
const std::string View3DGradientBackground::fs_radial_3color_shader_src = R"(
    varying vec2 uv;

    uniform vec3 from_color;
    uniform vec3 to_color;
    uniform vec3 mid_color;
    
    void main() {
        // Offset UV coordinates to center of viewport and scale them up slightly.
        // Without some scaling the `to_color` will be barely visible in the corners of the viewport.
        float scale_factor = 1.3;
        vec2 centered_uv = scale_factor * uv - (scale_factor * 0.5);
        float uv_length = length(centered_uv);

        // Convert incoming colors to linear color space for gradient calculations.
        vec3 from_color_linear = to_linear(from_color);
        vec3 to_color_linear = to_linear(to_color);
        vec3 mid_color_linear = to_linear(mid_color);

        vec3 fragment_color = oklab_mix(from_color_linear, mid_color_linear, smoothstep(0.0, 0.5, uv_length));
        fragment_color = oklab_mix(fragment_color, to_color_linear, smoothstep(0.5, 1.0, uv_length));

        fragment_color = dither(fragment_color, gl_FragCoord.xy);

        gl_FragColor = vec4(to_srgb(fragment_color), 1.0);
    }
)";
}  // namespace Gui
