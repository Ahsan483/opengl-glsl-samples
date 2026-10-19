"""Real-time OpenGL 3.3 core sample scenes rendered offscreen with moderngl.

render(scene, t, w, h) -> HxWx3 uint8. Scenes: shadows, textures, terrain, raymarch, particles.
"""
import math
import numpy as np
import moderngl

ctx = moderngl.create_standalone_context(require=330)

# ---------------------------------------------------------------- math

def perspective(fovy, aspect, near, far):
    f = 1 / math.tan(math.radians(fovy) / 2)
    m = np.zeros((4, 4), "f4")
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def ortho(l, r, b, t, n, f):
    m = np.eye(4, dtype="f4")
    m[0, 0], m[1, 1], m[2, 2] = 2 / (r - l), 2 / (t - b), -2 / (f - n)
    m[0, 3], m[1, 3], m[2, 3] = -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n)
    return m


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(lambda v: np.array(v, "f4"), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype="f4")
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def translate(x, y, z):
    m = np.eye(4, dtype="f4")
    m[:3, 3] = x, y, z
    return m


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], "f4")


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], "f4")


def scale(k):
    m = np.eye(4, dtype="f4")
    m[0, 0] = m[1, 1] = m[2, 2] = k
    return m


def gl(m):  # numpy row-major -> GLSL column-major bytes
    return np.ascontiguousarray(m.T, "f4").tobytes()


# ---------------------------------------------------------------- geometry (pos3 norm3 uv2 col3)

def _pack(pos, nrm, uv, col, idx):
    v = np.hstack([pos, nrm, uv, col]).astype("f4")
    return v[np.asarray(idx).ravel()]


def grid_mesh(nu, nv, fn):
    """fn(u, v) -> pos, normal arrays; u,v in [0,1]."""
    u, v = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1), indexing="ij")
    pos, nrm = fn(u, v)
    pos, nrm = pos.reshape(-1, 3), nrm.reshape(-1, 3)
    uv = np.stack([u.ravel(), v.ravel()], 1)
    idx = []
    for i in range(nu):
        for j in range(nv):
            a, b = i * (nv + 1) + j, (i + 1) * (nv + 1) + j
            idx += [a, b, a + 1, b, b + 1, a + 1]
    return _pack(pos, nrm, uv, np.ones_like(pos), idx)


def sphere(r=1.0, n=64):
    def fn(u, v):
        th, ph = u * 2 * np.pi, v * np.pi
        p = np.stack([np.sin(ph) * np.cos(th), np.cos(ph), np.sin(ph) * np.sin(th)], -1)
        return p * r, p
    return grid_mesh(n, n // 2, fn)


def torus(R=1.0, r=0.35, n=96, m=48):
    def fn(u, v):
        a, b = u * 2 * np.pi, v * 2 * np.pi
        c = np.stack([np.cos(a), np.zeros_like(a), np.sin(a)], -1)
        nrm = c * np.cos(b)[..., None] + np.array([0, 1, 0]) * np.sin(b)[..., None]
        return c * R + nrm * r, nrm
    return grid_mesh(n, m, fn)


def cube():
    faces = []
    for axis in range(3):
        for sgn in (-1, 1):
            n = np.zeros(3)
            n[axis] = sgn
            a, b = [(axis + 1) % 3, (axis + 2) % 3]
            corners = []
            for (s, t) in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                p = n.copy()
                p[a], p[b] = s * sgn, t
                corners.append((p, ((s + 1) / 2, (t + 1) / 2)))
            pos = np.array([c[0] for c in corners])
            uv = np.array([c[1] for c in corners])
            faces.append(_pack(pos, np.tile(n, (4, 1)), uv, np.ones((4, 3)), [0, 1, 2, 0, 2, 3]))
    return np.vstack(faces)


def plane(size=10):
    pos = np.array([[-size, 0, -size], [size, 0, -size], [size, 0, size], [-size, 0, size]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * size
    return _pack(pos, np.tile([0, 1, 0], (4, 1)), uv, np.ones((4, 3)), [0, 2, 1, 0, 3, 2])


def fbm(x, z, octaves=6):
    rng = np.random.default_rng(7)
    h = np.zeros_like(x)
    amp, freq = 1.0, 0.18
    for _ in range(octaves):
        px, pz = rng.uniform(0, 100, 2)
        h += amp * (np.sin((x + px) * freq) * np.cos((z + pz) * freq * 1.3)
                    + 0.5 * np.sin((x * 0.7 - z) * freq * 1.7 + px))
        amp *= 0.5
        freq *= 2.03
    return h


def terrain(n=260, size=40):
    xs = np.linspace(-size, size, n + 1)
    x, z = np.meshgrid(xs, xs, indexing="ij")
    h = fbm(x, z) * 3.2
    h = np.where(h < -1.2, -1.2 + (h + 1.2) * 0.05, h)  # flat lakes
    pos = np.stack([x, h, z], -1)
    gx, gz = np.gradient(h, xs, xs)
    nrm = np.stack([-gx, np.ones_like(h), -gz], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    t = (h - h.min()) / (h.max() - h.min())
    stops = [(0.0, (0.10, 0.32, 0.55)), (0.03, (0.16, 0.45, 0.66)), (0.08, (0.80, 0.74, 0.52)),
             (0.18, (0.30, 0.55, 0.22)), (0.50, (0.18, 0.40, 0.16)), (0.70, (0.45, 0.40, 0.35)),
             (0.85, (0.95, 0.96, 0.98)), (1.0, (1, 1, 1))]
    col = np.zeros(t.shape + (3,))
    for k in range(3):
        col[..., k] = np.interp(t, [s[0] for s in stops], [s[1][k] for s in stops])
    steep = 1 - nrm[..., 1]
    col = col * (1 - np.clip(steep * 2.5, 0, 0.6))[..., None] + np.array([0.42, 0.38, 0.34]) * np.clip(steep * 2.5, 0, 0.6)[..., None]
    idx = []
    for i in range(n):
        for j in range(n):
            a, b = i * (n + 1) + j, (i + 1) * (n + 1) + j
            idx += [a, a + 1, b, b, a + 1, b + 1]
    uv = np.zeros((pos.size // 3, 2))
    return _pack(pos.reshape(-1, 3), nrm.reshape(-1, 3), uv, col.reshape(-1, 3), idx)


def brick_texture(n=512):
    y, x = np.mgrid[0:n, 0:n] / n
    row = np.floor(y * 8)
    xx = x * 4 + (row % 2) * 0.5
    mortar = (np.abs((y * 8) % 1 - 0.5) > 0.44) | (np.abs(xx % 1 - 0.5) > 0.47)
    rng = np.random.default_rng(3)
    tint = rng.uniform(0.75, 1.1, (9, 6))[row.astype(int) % 9, np.floor(xx).astype(int) % 6]
    noise = rng.normal(1, 0.06, (n, n))
    base = np.stack([0.62, 0.26, 0.18]) * (tint * noise)[..., None]
    img = np.where(mortar[..., None], np.array([0.78, 0.76, 0.72]) * noise[..., None], base)
    return (img.clip(0, 1) * 255).astype("u1")


# ---------------------------------------------------------------- shaders

LIT_VS = """
#version 330 core
in vec3 in_pos; in vec3 in_norm; in vec2 in_uv; in vec3 in_col;
uniform mat4 model, view, proj, light_vp;
out vec3 v_pos; out vec3 v_norm; out vec2 v_uv; out vec3 v_col; out vec4 v_lpos;
void main() {
    vec4 wp = model * vec4(in_pos, 1.0);
    v_pos = wp.xyz;
    v_norm = mat3(transpose(inverse(model))) * in_norm;
    v_uv = in_uv; v_col = in_col;
    v_lpos = light_vp * wp;
    gl_Position = proj * view * wp;
}
"""

LIT_FS = """
#version 330 core
in vec3 v_pos; in vec3 v_norm; in vec2 v_uv; in vec3 v_col; in vec4 v_lpos;
uniform vec3 light_dir, cam_pos, tint, sky, fog_col;
uniform float shininess, spec_k, fog_density;
uniform int use_tex, checker, use_shadow;
uniform sampler2D tex; uniform sampler2D shadow_map;
out vec4 frag;

float shadow() {
    if (use_shadow == 0) return 1.0;
    vec3 p = v_lpos.xyz / v_lpos.w * 0.5 + 0.5;
    if (p.z > 1.0 || any(lessThan(p.xy, vec2(0.0))) || any(greaterThan(p.xy, vec2(1.0)))) return 1.0;
    float s = 0.0; vec2 texel = 1.0 / vec2(textureSize(shadow_map, 0));
    for (int x = -2; x <= 2; ++x) for (int y = -2; y <= 2; ++y)
        s += (p.z - 0.0015 > texture(shadow_map, p.xy + vec2(x, y) * texel).r) ? 0.0 : 1.0;
    return s / 25.0;
}

void main() {
    vec3 n = normalize(v_norm);
    vec3 base = v_col * tint;
    if (use_tex == 1) base *= texture(tex, v_uv).rgb;
    if (checker == 1) {
        float c = mod(floor(v_uv.x) + floor(v_uv.y), 2.0);
        base *= mix(0.55, 0.85, c);
    }
    vec3 l = normalize(light_dir), v = normalize(cam_pos - v_pos), h = normalize(l + v);
    float diff = max(dot(n, l), 0.0);
    float spec = pow(max(dot(n, h), 0.0), shininess) * spec_k;
    float sh = shadow();
    vec3 ambient = base * mix(vec3(0.10, 0.11, 0.14), sky * 0.35, n.y * 0.5 + 0.5);
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.25;
    vec3 col = ambient + sh * (base * diff * vec3(1.0, 0.96, 0.9) + spec) + rim * sky * 0.4;
    float d = length(cam_pos - v_pos);
    col = mix(col, fog_col, 1.0 - exp(-d * fog_density));
    frag = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}
"""

DEPTH_VS = """
#version 330 core
in vec3 in_pos; uniform mat4 model, light_vp;
void main() { gl_Position = light_vp * model * vec4(in_pos, 1.0); }
"""
DEPTH_FS = "#version 330 core\nvoid main() {}\n"

QUAD_VS = """
#version 330 core
in vec2 in_xy; out vec2 uv;
void main() { uv = in_xy * 0.5 + 0.5; gl_Position = vec4(in_xy, 0.0, 1.0); }
"""

SKY_FS = """
#version 330 core
in vec2 uv; uniform vec3 top, bottom; out vec4 frag;
void main() { frag = vec4(pow(mix(bottom, top, pow(uv.y, 0.8)), vec3(1.0/2.2)), 1.0); }
"""

# Raymarched signed-distance scene: soft shadows, ambient occlusion, reflections.
RAY_FS = """
#version 330 core
in vec2 uv; uniform float time; uniform vec2 res; out vec4 frag;

float sdBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float sdTorus(vec3 p, vec2 t) { vec2 q = vec2(length(p.xz) - t.x, p.y); return length(q) - t.y; }
float smin(float a, float b, float k) { float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0); return mix(b, a, h) - k * h * (1.0 - h); }
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

vec2 map(vec3 p) {
    float ground = p.y + 1.0;
    vec3 q = p - vec3(0.0, 0.2, 0.0);
    q.xz *= rot(time * 0.6); q.xy *= rot(time * 0.4);
    float blob = smin(length(q) - 0.75, sdTorus(q, vec2(1.0, 0.22)), 0.35);
    float box = sdBox(p - vec3(-2.2, -0.45, 0.6), vec3(0.5)) - 0.06;
    float ball = length(p - vec3(2.1, -0.35 + 0.25 * sin(time * 2.0), 0.4)) - 0.65;
    vec2 r = vec2(ground, 1.0);
    if (blob < r.x) r = vec2(blob, 2.0);
    if (box < r.x) r = vec2(box, 3.0);
    if (ball < r.x) r = vec2(ball, 4.0);
    return r;
}
vec3 normal(vec3 p) { vec2 e = vec2(0.001, 0.0); return normalize(vec3(map(p + e.xyy).x - map(p - e.xyy).x, map(p + e.yxy).x - map(p - e.yxy).x, map(p + e.yyx).x - map(p - e.yyx).x)); }
vec2 march(vec3 ro, vec3 rd) { float t = 0.0; for (int i = 0; i < 160; i++) { vec2 h = map(ro + rd * t); if (h.x < 0.0005 * t) return vec2(t, h.y); t += h.x; if (t > 40.0) break; } return vec2(-1.0); }
float softShadow(vec3 ro, vec3 rd) { float res = 1.0, t = 0.02; for (int i = 0; i < 64; i++) { float h = map(ro + rd * t).x; res = min(res, 10.0 * h / t); t += clamp(h, 0.02, 0.3); if (res < 0.001 || t > 12.0) break; } return clamp(res, 0.0, 1.0); }
float ao(vec3 p, vec3 n) { float o = 0.0, s = 1.0; for (int i = 1; i <= 5; i++) { float h = 0.03 * float(i); o += (h - map(p + n * h).x) * s; s *= 0.8; } return clamp(1.0 - 3.0 * o, 0.0, 1.0); }
vec3 material(float id, vec3 p) {
    if (id < 1.5) { float c = mod(floor(p.x) + floor(p.z), 2.0); return mix(vec3(0.12), vec3(0.32), c); }
    if (id < 2.5) return vec3(0.15, 0.65, 0.95);
    if (id < 3.5) return vec3(0.95, 0.55, 0.15);
    return vec3(0.85, 0.2, 0.45);
}
vec3 shade(vec3 ro, vec3 rd, out vec3 hitP, out vec3 hitN, out float hitId) {
    vec3 sky = mix(vec3(0.02, 0.03, 0.08), vec3(0.15, 0.25, 0.5), clamp(rd.y * 1.5 + 0.4, 0.0, 1.0));
    vec2 h = march(ro, rd); hitId = -1.0;
    if (h.x < 0.0) return sky;
    vec3 p = ro + rd * h.x, n = normal(p);
    hitP = p; hitN = n; hitId = h.y;
    vec3 l = normalize(vec3(0.6, 0.8, -0.4));
    vec3 m = material(h.y, p);
    float dif = max(dot(n, l), 0.0) * softShadow(p + n * 0.01, l);
    float spe = pow(max(dot(reflect(-l, n), -rd), 0.0), 48.0) * dif;
    float a = ao(p, n);
    vec3 col = m * (0.18 * a + 1.1 * dif * vec3(1.0, 0.95, 0.85)) + 0.6 * spe;
    col += m * 0.15 * a * vec3(0.3, 0.5, 1.0) * clamp(0.5 + 0.5 * n.y, 0.0, 1.0);
    return mix(col, sky, 1.0 - exp(-0.0025 * h.x * h.x));
}
void main() {
    vec2 p = (uv * res - 0.5 * res) / res.y;
    float a = 0.25 * time;
    vec3 ro = vec3(5.2 * sin(a), 1.6, -5.2 * cos(a)), ta = vec3(0.0, -0.1, 0.0);
    vec3 f = normalize(ta - ro), r = normalize(cross(f, vec3(0, 1, 0))), u = cross(r, f);
    vec3 rd = normalize(p.x * r + p.y * u + 1.8 * f);
    vec3 hp, hn; float id;
    vec3 col = shade(ro, rd, hp, hn, id);
    if (id > 0.0 && id != 3.0) {  // one reflection bounce
        vec3 hp2, hn2; float id2;
        float k = id < 1.5 ? 0.25 : 0.35;
        col = mix(col, shade(hp + hn * 0.01, reflect(rd, hn), hp2, hn2, id2), k);
    }
    col = pow(col, vec3(0.4545));
    col *= 1.0 - 0.25 * dot(p, p);
    frag = vec4(col, 1.0);
}
"""

PART_VS = """
#version 330 core
in vec3 in_pos; in vec3 in_col; in float in_size;
uniform mat4 view, proj; out vec3 v_col;
void main() {
    vec4 vp = view * vec4(in_pos, 1.0);
    gl_Position = proj * vp;
    gl_PointSize = in_size * 900.0 / -vp.z;
    v_col = in_col;
}
"""
PART_FS = """
#version 330 core
in vec3 v_col; out vec4 frag;
void main() {
    vec2 d = gl_PointCoord - 0.5; float r = dot(d, d) * 4.0;
    if (r > 1.0) discard;
    frag = vec4(v_col * exp(-r * 3.0), 1.0);
}
"""

lit = ctx.program(vertex_shader=LIT_VS, fragment_shader=LIT_FS)
depth = ctx.program(vertex_shader=DEPTH_VS, fragment_shader=DEPTH_FS)
skyp = ctx.program(vertex_shader=QUAD_VS, fragment_shader=SKY_FS)
ray = ctx.program(vertex_shader=QUAD_VS, fragment_shader=RAY_FS)
part = ctx.program(vertex_shader=PART_VS, fragment_shader=PART_FS)

quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
sky_vao = ctx.vertex_array(skyp, [(quad, "2f", "in_xy")])
ray_vao = ctx.vertex_array(ray, [(quad, "2f", "in_xy")])


class Mesh:
    def __init__(self, data):
        self.vbo = ctx.buffer(data.tobytes())
        self.vao = ctx.vertex_array(lit, [(self.vbo, "3f 3f 2f 3f", "in_pos", "in_norm", "in_uv", "in_col")])
        self.dvao = ctx.vertex_array(depth, [(self.vbo, "3f 12x 8x 12x", "in_pos")])


MESH = dict(sphere=Mesh(sphere()), torus=Mesh(torus()), cube=Mesh(cube()), plane=Mesh(plane(60)),
            terrain=Mesh(terrain()))
brick = ctx.texture((512, 512), 3, brick_texture().tobytes())
brick.build_mipmaps()
brick.anisotropy = 8.0
brick.repeat_x = brick.repeat_y = True

SHADOW = ctx.depth_texture((2048, 2048))
SHADOW.compare_func = ""
SHADOW.repeat_x = SHADOW.repeat_y = False
shadow_fbo = ctx.framebuffer(depth_attachment=SHADOW)
_fbos = {}


def _targets(w, h):
    if (w, h) not in _fbos:
        ms = ctx.framebuffer(ctx.renderbuffer((w, h), 4, samples=8), ctx.depth_renderbuffer((w, h), samples=8))
        out = ctx.framebuffer(ctx.renderbuffer((w, h), 4))
        _fbos[(w, h)] = (ms, out)
    return _fbos[(w, h)]


def _set(prog, **u):
    for k, v in u.items():
        if k in prog:
            prog[k].write(gl(v)) if isinstance(v, np.ndarray) and v.shape == (4, 4) else setattr(prog[k], "value", v)


def _sky(top, bottom):
    ctx.disable(moderngl.DEPTH_TEST)
    skyp["top"].value, skyp["bottom"].value = top, bottom
    sky_vao.render(moderngl.TRIANGLE_STRIP)
    ctx.enable(moderngl.DEPTH_TEST)


def _draw(objs, view, proj, cam, light_dir, light_vp, sky, fog_col=(0.05, 0.06, 0.1), fog=0.0, shadows=True):
    if shadows:
        shadow_fbo.use()
        shadow_fbo.clear(depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        for name, model, *_ in objs:
            depth["model"].write(gl(model))
            depth["light_vp"].write(gl(light_vp))
            MESH[name].dvao.render()
    return objs


def _scene_lit(t, w, h, objs, cam, target, light_dir, sky_top, sky_bot, fog=0.0, fog_col=(0, 0, 0), fov=45, shadows=True):
    view = look_at(cam, target)
    proj = perspective(fov, w / h, 0.1, 200)
    ld = np.array(light_dir, "f4") / np.linalg.norm(light_dir)
    light_vp = ortho(-14, 14, -14, 14, 0.1, 50) @ look_at(ld * 18, (0, 0, 0))
    _draw(objs, view, proj, cam, ld, light_vp, sky_top, shadows=shadows)
    ms, out = _targets(w, h)
    ms.use()
    ms.clear(0, 0, 0, 1)
    if fog > 0.03:  # dense fog: background must match the fog so the horizon melts away
        _sky(tuple(min(1, c * 1.8) for c in fog_col), fog_col)
    else:
        _sky(sky_top, sky_bot)
    SHADOW.use(1)
    brick.use(0)
    lit["tex"].value, lit["shadow_map"].value = 0, 1
    for name, model, tint, tex, chk, shin, speck in objs:
        _set(lit, model=model, view=view, proj=proj, light_vp=light_vp)
        lit["light_dir"].value = tuple(ld)
        lit["cam_pos"].value = tuple(cam)
        lit["tint"].value, lit["sky"].value, lit["fog_col"].value = tint, sky_top, fog_col
        lit["shininess"].value, lit["spec_k"].value, lit["fog_density"].value = shin, speck, fog
        lit["use_tex"].value, lit["checker"].value, lit["use_shadow"].value = tex, chk, int(shadows)
        MESH[name].vao.render()
    ctx.copy_framebuffer(out, ms)
    return out


def _read(fbo, w, h):
    img = np.frombuffer(fbo.read(components=3), "u1").reshape(h, w, 3)
    return img[::-1].copy()


def render(scene, t, w=1280, h=720):
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    ctx.disable(moderngl.BLEND)
    if scene == "shadows":
        objs = [
            ("plane", np.eye(4, dtype="f4"), (0.55, 0.57, 0.66), 0, 1, 16.0, 0.15),
            ("torus", translate(0, 1.15, 0) @ rot_y(t * 0.7) @ rot_x(0.9 + 0.3 * math.sin(t)), (0.20, 0.65, 1.0), 0, 0, 64.0, 0.9),
            ("sphere", translate(-2.4, 0.8, 1.2) @ scale(0.8), (1.0, 0.45, 0.25), 0, 0, 96.0, 1.0),
            ("cube", translate(2.4, 0.6 + 0.15 * math.sin(t * 2), 0.8) @ rot_y(-t * 0.9) @ scale(0.6), (0.35, 0.85, 0.5), 0, 0, 32.0, 0.5),
        ]
        a = 0.35 + t * 0.15
        cam = (7.5 * math.sin(a), 4.2, 7.5 * math.cos(a))
        out = _scene_lit(t, w, h, objs, cam, (0, 0.7, 0), (0.6, 1.0, 0.35), (0.06, 0.09, 0.2), (0.01, 0.015, 0.04), fog=0.07, fog_col=(0.012, 0.018, 0.045))
    elif scene == "textures":
        objs = [("plane", translate(0, -1.4, 0), (0.45, 0.42, 0.40), 0, 1, 8.0, 0.1)]
        for k, (x, z) in enumerate([(-2.6, 0), (0, -0.6), (2.6, 0)]):
            m = translate(x, 0.05 * math.sin(t * 2 + k), z) @ rot_y(t * 0.8 + k) @ rot_x(0.5 + 0.3 * k)
            objs.append(("cube", m, (1.0, 1.0, 1.0), 1, 0, 24.0, 0.25))
        cam = (0.0 + 1.2 * math.sin(t * 0.3), 2.6, 7.0)
        out = _scene_lit(t, w, h, objs, cam, (0, -0.1, 0), (-0.5, 1.0, 0.7), (0.22, 0.15, 0.08), (0.03, 0.02, 0.01), fog=0.08, fog_col=(0.025, 0.018, 0.012))
    elif scene == "terrain":
        objs = [("terrain", np.eye(4, dtype="f4"), (1, 1, 1), 0, 0, 8.0, 0.08)]
        z = 30 - t * 3.2
        cam = (6 * math.sin(t * 0.25), 9.0, z)
        out = _scene_lit(t, w, h, objs, cam, (6 * math.sin(t * 0.25 + 0.4), 2.0, z - 14), (0.5, 0.45, -0.75),
                         (0.30, 0.48, 0.80), (0.80, 0.72, 0.62), fog=0.010, fog_col=(0.62, 0.66, 0.76), fov=55, shadows=False)
    elif scene == "raymarch":
        ms, out = _targets(w, h)
        ms.use()
        ctx.disable(moderngl.DEPTH_TEST)
        ray["time"].value, ray["res"].value = t, (w, h)
        ray_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.copy_framebuffer(out, ms)
    elif scene == "particles":
        ms, out = _targets(w, h)
        ms.use()
        ms.clear(0.01, 0.01, 0.03, 1)
        _sky((0.04, 0.03, 0.10), (0.0, 0.0, 0.02))
        rng = np.random.default_rng(11)
        n = 22000
        life = 3.2
        born = rng.uniform(0, life, n)
        age = (t + born) % life
        ang = rng.uniform(0, 2 * np.pi, n)
        spd = rng.uniform(1.0, 3.2, n)
        up = rng.uniform(6.0, 9.5, n)
        swirl = ang + age * 0.8
        pos = np.stack([np.cos(swirl) * spd * age, up * age - 4.9 * age ** 2 * 0.9, np.sin(swirl) * spd * age], 1)
        pos[:, 1] = np.abs(pos[:, 1])  # bounce on the floor
        k = age / life
        hue = rng.integers(0, 3, n)
        pal = np.array([[0.2, 0.75, 1.0], [1.0, 0.3, 0.75], [1.0, 0.65, 0.15]])
        col = (pal[hue] * (1 - k)[:, None] + np.array([1.0, 0.95, 0.9]) * np.clip(0.25 - k, 0, 1)[:, None] * 3) * (1 - k)[:, None] * 0.3
        size = 0.06 + 0.08 * (1 - k)
        data = np.hstack([pos, col, size[:, None]]).astype("f4")
        vbo = ctx.buffer(data.tobytes())
        vao = ctx.vertex_array(part, [(vbo, "3f 3f 1f", "in_pos", "in_col", "in_size")])
        a = t * 0.2
        view = look_at((7.5 * math.sin(a), 3.0, 7.5 * math.cos(a)), (0, 3.4, 0))
        _set(part, view=view, proj=perspective(50, w / h, 0.1, 100))
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        vao.render(moderngl.POINTS)
        ctx.disable(moderngl.BLEND)
        ctx.copy_framebuffer(out, ms)
        vao.release()
        vbo.release()
    return _read(out, w, h)


if __name__ == "__main__":
    import os
    from PIL import Image
    os.makedirs("renders", exist_ok=True)
    for s, t in [("shadows", 1.0), ("textures", 1.2), ("terrain", 2.0), ("raymarch", 1.3), ("particles", 2.2)]:
        Image.fromarray(render(s, t, 1600, 900)).save(f"renders/{s}.jpg")
        print("ok", s)
