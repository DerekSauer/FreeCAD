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

#ifndef GUI_VIEW3DBACKGROUND_H
#define GUI_VIEW3DBACKGROUND_H

#include <optional>
#include <QtOpenGL>
#include <QColor>

namespace Gui
{

class View3DInventorViewer;

/// Renders smooth gradients to the background of an Inventor viewer.
class View3DGradientBackground
{
public:
    View3DGradientBackground(View3DInventorViewer* viewer);
    ~View3DGradientBackground();

    void set_gradient_colors(const QColor& from_color,
                             const QColor& to_color,
                             const std::optional<QColor>& mid_color = std::nullopt);

    void draw_linear_gradient();
    void draw_radial_gradient();

private:
    /// Type of shader being generated by `create_shader()`.
    enum ShaderType
    {
        VertexShader = GL_VERTEX_SHADER,

        FragmentShader = GL_FRAGMENT_SHADER
    };

    void prepare(QOpenGLFunctions* gl);
    std::optional<QOpenGLFunctions*> gl_functions();
    GLuint compile_shader(QOpenGLFunctions* gl,
                          ShaderType shader_type,
                          const std::vector<std::string>& shader_source);
    GLuint link_shader_program(QOpenGLFunctions* gl, const std::vector<GLuint>& compiled_shaders);

private:
    /// Pointer to the viewer that will use this background gradient.
    /// SAFETY: The viewer is constructed before this object. This object is destructed before
    /// the viewer, hence this pointer should remain valid for the lifetime of the viewer.
    View3DInventorViewer* viewer;

    /// Start color for the gradient.
    QColor from_color {"magenta"};

    /// End color for the gradient.
    QColor to_color {"cyan"};

    /// Optional middle color for the gradient.
    std::optional<QColor> mid_color;

    /// Angle of linear gradients in degrees:
    ///  0.0° = `from_color` to `to_color` to the right.
    /// 90.0° = `from_color` to `to_color` upwards.
    float angle = 270.0F;

private:
    /// Flag indicating if the OpenGL functionality needed to render a gradient background was
    /// successfully initialized. If `false` the platform does not support necessary OpenGL
    /// features (or ran out of VRAM); a fallback background rendering method will be used.
    bool gl_init_success = false;

    /// Flag indicating if this object is ready to render gradient backgrounds.
    /// It is necessary to prepare for rendering `just-in-time` as a GL context does not exist until
    /// after the viewer has been initialized.
    bool prepared_to_draw = false;

    /// Vertex buffer containing the vertices and UV-coordinates to render
    /// a quad that covers the entire viewport.
    GLuint quad_vertex_buffer = 0;

    /// Shader program rendering a two color linear gradient.
    GLuint linear_2color_gradient_program = 0;

    /// Shader program rendering a three color linear gradient.
    GLuint linear_3color_gradient_program = 0;

    /// Shader program rendering a two color radial gradient.
    GLuint radial_2color_gradient_program = 0;

    /// Shader program rendering a three color radial gradient.
    GLuint radial_3color_gradient_program = 0;

private:
    /// Shader attribute slot bound to the position of vertices passed to the shader.
    static const GLuint vertex_coord_attribute_slot;

    /// Shader attribute slot bound to the uv-coord of vertices passed to the shader.
    static const GLuint uv_coord_attribute_slot;

    /// Name of the shader uniform passing the start color of the gradient to the shader.
    static const std::string from_color_uniform;

    /// Name of the shader uniform passing the end color of the gradient to the shader.
    static const std::string to_color_uniform;

    /// Name of the shader uniform passing the middle color of the gradient to the shader.
    static const std::string mid_color_uniform;

    /// Name of the shader uniform passing the angle of a linear gradient to the shader.
    static const std::string angle_uniform;

    /// Name of the shader uniform passing the midpoint location of the gradient to the shader.
    static const std::string midpoint_bias_uniform;

    static const std::string vs_shader_src;
    static const std::string fs_common_src;
    static const std::string fs_linear_2color_shader_src;
    static const std::string fs_linear_3color_shader_src;
    static const std::string fs_radial_2color_shader_src;
    static const std::string fs_radial_3color_shader_src;
};

}  // namespace Gui

#endif  // GUI_VIEW3DBACKGROUND_H
