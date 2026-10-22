"""
malha3d.py - transforma a nuvem de pontos de um varrimento numa superfície ("método A"): os pontos
laterais são unidos numa banda fechada à volta do objeto (por agora aberta em cima e em baixo).

Utilização:
    python malha3d.py caixa12.txt                 # último varrimento completo do ficheiro
    python malha3d.py largo_H155.txt largo_H12.txt   # vários logs do mesmo objeto, juntos
    python malha3d.py caixa12.txt --dtheta 5 --dz 0.5

Precisa do scan3d.py na mesma pasta (para ler os logs e posicionar os pontos), de numpy,
matplotlib e (para a vista interativa) plotly.

Saídas (ao lado do primeiro log):
    _malha.html   superfície 3D interativa + pontos
    _malha.png    vistas estáticas
    _malha.ply    a superfície como malha (vértices + triângulos), abre no MeshLab

Como se constrói a superfície:
  1. Os pontos são posicionados como no scan3d.py (sobre o eixo do feixe), os ecos do prato são
     removidos e as leituras repetidas juntas.
  2. O eixo do objeto é encontrado por um ajuste de circunferência aos pontos vistos de cima, e
     cada ponto é escrito como (ângulo à volta do eixo theta, altura z, distância ao eixo r).
     Com vários logs, cada um é centrado no seu próprio eixo (o objeto pode ter sido pousado em
     sítios ligeiramente diferentes).
  3. Coloca-se uma grelha de nós ângulo x altura à volta do eixo; cada nó toma uma média pesada
     robusta dos pontos próximos: pesos gaussianos na distância (em ângulo e em altura) ao nó, e
     pesos de Cauchy que diminuem a influência dos pontos longe da superfície local, para que
     ecos perdidos não a puxem. Os pesos robustos são recalculados em 3 passagens (mínimos
     quadrados iterativamente re-pesados).
  4. Os nós com poucos pontos por perto ("peso" < MIN_WEIGHT) tomam uma média mais larga
     (núcleo 3 vezes maior) e são desenhados a cinzento: são uma estimativa por interpolação,
     não uma medição.
  5. Nós vizinhos são unidos em triângulos, fechando a volta completa (360 graus).

A superfície fica onde os pontos estão, por isso herda o seu viés para fora (cerca de +0,8 a 1 cm):
o HC-SR04 devolve o ponto MAIS PRÓXIMO dentro do cone do feixe (semi-abertura ~12-13 graus), mas o
ponto é colocado no EIXO do feixe; num objeto convexo isso desloca-o radialmente para fora. A
banda mostra a forma; as dimensões com incerteza vêm de dimensoes.py / prisma.py, e o
volume_banda.py corrige este viés com uma calibração na lata.

Limitações: a superfície é uma função r(theta, z), portanto só descreve objetos "em estrela" em
relação ao eixo (cada semirreta horizontal a partir do eixo corta a superfície uma só vez);
cavidades e reentrâncias não são representadas. Topo e base não são fechados.
"""

import argparse
import os

import numpy as np
import matplotlib
# Backend sem janela: as figuras vão para ficheiro.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scan3d

# ========================= CONFIGURAÇÕES =========================
# Grelha: o passo deve ser menor do que o alcance dos núcleos (SIGMA_*) para a superfície ficar
# suave; diminuir dá mais detalhe aparente mas não mais resolução real (o feixe é largo).
DTHETA = 5.0        # graus, passo da grelha à volta do eixo (= passo do prato nos varrimentos)
DZ = 0.5            # cm, passo da grelha em altura
# Alcance da influência de cada ponto (desvio-padrão do peso gaussiano). Maior = mais suave e
# menos nós "preenchidos", mas arredonda arestas; menor = mais detalhe mas mais buracos.
SIGMA_THETA = 10.0  # graus, alcance à volta do eixo
SIGMA_Z = 1.0       # cm, alcance para cima/para baixo
# Escala do peso robusto (Cauchy): um ponto a ROBUST_R da superfície local pesa 1/2, a 2*ROBUST_R
# pesa 1/5. Da ordem da dispersão normal dos pontos (resolução ~0,33 cm + ruído).
ROBUST_R = 0.5      # cm
# "Número equivalente de pontos" (soma dos pesos) que um nó precisa para contar como medido.
MIN_WEIGHT = 1.5
MAX_RADIUS = 7.0    # cm, pontos mais longe do que isto do eixo do prato são fundo (rejeitados);
                    # aumentar para objetos com mais de ~6 cm de raio
Z_MIN = 1.0         # cm, pontos mais baixos do que isto são ecos do prato
Z_TRIM = (2, 98)    # percentis das alturas dos pontos que definem a extensão vertical da banda
                    # (cortar 2 % em cima e em baixo evita esticá-la por pontos isolados)
# =================================================================


def load_points(path):
    """Lê um log e devolve os pontos do último varrimento completo, já filtrados.

    Entrada: path - ficheiro de log do Arduino.
    Saída: (P, R, H)
        P - array (N, 7) no formato de scan3d.to_points (x, y, z em cm no referencial do objeto,
            ângulo do prato, inclinação, distância, repetições);
        R, H - geometria do sensor lida do log (cm).
    Usa as funções do scan3d (mesma leitura e mesma geometria) mas com filtros próprios: só
    leituras com eco, repetições sempre juntas, z >= Z_MIN (sem prato) e distância ao eixo do
    prato <= MAX_RADIUS (sem fundo). Se nenhum varrimento estiver completo usa o último que houver.
    """
    scans = scan3d.parse_log(path)
    complete = [s for s in scans if s["complete"]] or scans
    s = complete[-1]
    d = s["rows"]
    # Só leituras com eco (distância > 0).
    d = d[d[:, 3] > 0]
    d = scan3d.merge_repeats(d)
    P = scan3d.to_points(d, s["R"], s["H"])
    # Remover ecos do prato (z baixo) e do fundo (longe do eixo).
    P = P[(P[:, 2] >= Z_MIN) & (np.hypot(P[:, 0], P[:, 1]) <= MAX_RADIUS)]
    return P, s["R"], s["H"]


def circle_centre(x, y):
    """Centro da circunferência que melhor se ajusta aos pontos (x, y), em cm.

    Ajuste algébrico linear (método de Kasa): a circunferência (x-cx)^2 + (y-cy)^2 = r^2 reescreve-se
    como x^2 + y^2 = 2 cx x + 2 cy y + c, com c = r^2 - cx^2 - cy^2, que é LINEAR nas incógnitas
    (cx, cy, c). Resolve-se por mínimos quadrados sem iterações nem valor inicial. Só o centro
    interessa (c é descartado). Serve para encontrar o eixo do objeto, que não coincide exatamente
    com o eixo do prato. Para objetos não circulares (p.ex. caixas) dá um centro aproximado, o que
    chega para definir o eixo da banda.
    """
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    cx, cy, _ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)[0]
    return cx, cy


def fill_periodic(row):
    """Interpolação linear dos NaN numa linha periódica (p.ex. uma volta de 360 graus).

    Entrada: row - array 1D com NaN nos valores em falta; o último elemento é vizinho do primeiro.
    Saída: array 1D sem NaN (se não houver nenhum valor válido devolve a linha tal como está; se
    houver só um, a linha toda toma esse valor).
    Ideia: replica-se a linha três vezes (deslocada de -n, 0, +n) para que a interpolação atravesse
    a costura 360 -> 0 graus como se a linha fosse contínua.
    (Função auxiliar; não é usada no fluxo principal deste script.)
    """
    n = len(row)
    ok = ~np.isnan(row)
    if ok.sum() == 0:
        return row
    if ok.sum() == 1:
        return np.full(n, row[ok][0])
    idx = np.arange(n)
    xs, ys = idx[ok], row[ok]
    xs = np.r_[xs - n, xs, xs + n]
    ys = np.r_[ys, ys, ys]
    return np.interp(idx, xs, ys)


def _kernel_average(th_c, z_c, th, z, r, w_pt, s_th, s_z):
    """Média pesada de r à volta de cada nó da grelha (o ângulo dá a volta).

    Entradas:
        th_c (nth,) - ângulos dos nós (rad);  z_c (nz,) - alturas dos nós (cm);
        th, z, r (N,) - ângulo (rad), altura (cm) e distância ao eixo (cm) de cada ponto;
        w_pt (N,) - peso próprio de cada ponto (peso robusto, entre 0 e 1);
        s_th (rad), s_z (cm) - larguras dos núcleos gaussianos.
    Saídas:
        média (nz, nth) - r médio em cada nó (cm);
        sw (nz, nth)    - soma dos pesos em cada nó = "número equivalente de pontos" que o
                          suportam (um ponto mesmo em cima do nó e com peso robusto 1 conta 1).
    Peso de um ponto num nó: exp(-1/2 [(dtheta/s_th)^2 + (dz/s_z)^2]) * w_pt (gaussiana 2D).
    A diferença de ângulos é reduzida a ]-pi, pi] com angle(exp(i*dtheta)), para que um ponto a
    179 graus e um nó a -179 graus fiquem a 2 graus e não a 358.
    Cálculo vetorizado com broadcasting num array (nz, nth, N): rápido, mas a memória cresce com
    nz*nth*N.
    """
    dth = np.angle(np.exp(1j * (th_c[None, :, None] - th[None, None, :])))      # (1, nth, N)
    dzz = z_c[:, None, None] - z[None, None, :]                                 # (nz, 1, N)
    W = np.exp(-0.5 * ((dth / s_th) ** 2 + (dzz / s_z) ** 2)) * w_pt[None, None, :]
    sw = W.sum(axis=2)
    # O maximum(.., 1e-12) evita dividir por zero em nós sem nenhum ponto perto.
    return (W * r[None, None, :]).sum(axis=2) / np.maximum(sw, 1e-12), sw


def build_band(th, z, r, dtheta, dz):
    """Superfície lateral suave r(ângulo, altura) a partir dos pontos.

    Entradas:
        th, z, r - ângulo à volta do eixo do objeto (rad), altura (cm) e distância ao eixo (cm)
                   de cada ponto;
        dtheta (graus), dz (cm) - passos da grelha.
    Saídas: (th_c, z_c, rad, measured, sw)
        th_c (nth,) - ângulos dos nós (rad), centrados nos intervalos de -180 a 180 graus;
        z_c (nz,)   - alturas dos nós (cm), entre os percentis Z_TRIM das alturas dos pontos;
        rad (nz, nth) - distância da superfície ao eixo em cada nó (cm);
        measured (nz, nth) - True nos nós com dados suficientes (sw >= MIN_WEIGHT);
        sw (nz, nth) - soma dos pesos em cada nó.

    Cada nó toma uma média pesada robusta dos pontos perto dele (pesos gaussianos,
    SIGMA_THETA x SIGMA_Z); os pontos longe da superfície local perdem peso (Cauchy, ROBUST_R)
    ao longo de algumas passagens. Os nós com menos de MIN_WEIGHT pontos equivalentes de dados
    são "preenchidos": tomam uma média mais larga e são desenhados a cinzento.

    Porquê robusto: a média simples é puxada por ecos falsos (p.ex. leituras "longas demais"
    quando o ponto mais próximo é oblíquo e não responde, ou ecos de outra parte do objeto). O peso
    de Cauchy 1/(1 + (res/c)^2) quase não mexe em pontos com resíduo pequeno e anula os muito
    afastados, sem um corte abrupto. É um estimador M (mínimos quadrados iterativamente re-pesados).
    """
    # Nós em ângulo: centros dos intervalos [-180, -180+dtheta[, ... (em rad).
    th_c = np.radians(np.arange(-180, 180, dtheta) + dtheta / 2)
    # Extensão vertical: do percentil 2 ao 98 das alturas (ignora pontos isolados nos extremos).
    z_lo, z_hi = np.percentile(z, Z_TRIM)
    z_c = np.arange(z_lo, z_hi + 1e-9, dz)
    s_th, s_z = np.radians(SIGMA_THETA), SIGMA_Z

    # Começa com todos os pontos de peso 1; 3 passagens de re-pesagem.
    w_pt = np.ones(len(r))
    for _ in range(3):
        rad, sw = _kernel_average(th_c, z_c, th, z, r, w_pt, s_th, s_z)
        # Valor da superfície em cada ponto (nó vizinho) -> resíduo -> peso robusto.
        # Ângulo: o nó mais próximo (diferença angular reduzida a ]-pi, pi]).
        # Altura: searchsorted dá o primeiro nó acima do ponto (aproximação ao nó mais próximo,
        # erro de no máximo um passo DZ); clip para não sair da grelha.
        jt = np.argmin(np.abs(np.angle(np.exp(1j * (th[:, None] - th_c[None, :])))), axis=1)
        iz = np.clip(np.searchsorted(z_c, z), 0, len(z_c) - 1)
        res = r - rad[iz, jt]
        # Peso de Cauchy: 1 para resíduo 0, 1/2 para |res| = ROBUST_R, -> 0 para resíduos grandes.
        w_pt = 1.0 / (1.0 + (res / ROBUST_R) ** 2)

    # Nós medidos: os que têm pelo menos MIN_WEIGHT pontos equivalentes (sw da última média).
    measured = sw >= MIN_WEIGHT
    # Nós sem dados suficientes: média com núcleos 3 vezes mais largos (com os pesos robustos
    # finais). Isto é uma interpolação/extrapolação, por isso esses nós ficam marcados.
    wide, _ = _kernel_average(th_c, z_c, th, z, r, w_pt, 3 * s_th, 3 * s_z)
    rad = np.where(measured, rad, wide)
    return th_c, z_c, rad, measured, sw


def band_mesh(th_c, z_c, rad):
    """Converte a grelha r(theta, z) numa malha de triângulos.

    Entradas: th_c (nth,) ângulos (rad), z_c (nz,) alturas (cm), rad (nz, nth) raios (cm).
    Saídas:
        V (nz*nth, 3) - vértices x = r cos(theta), y = r sin(theta), z (cm); o vértice do nó
                        (i, j) tem o índice i*nth + j;
        F (2*(nz-1)*nth, 3) - triângulos, como índices de vértices.
    Cada quadrilátero entre as alturas i, i+1 e os ângulos j, j+1 é dividido em dois triângulos.
    O índice angular (j+1) % nth faz o último ângulo ligar ao primeiro: a banda fecha à volta.
    """
    nz, nth = rad.shape
    TH, Z = np.meshgrid(th_c, z_c)
    V = np.c_[(rad * np.cos(TH)).ravel(), (rad * np.sin(TH)).ravel(), Z.ravel()]
    F = []
    for i in range(nz - 1):
        for j in range(nth):
            # Cantos do quadrilátero: a, b em baixo (altura i), c, d em cima (altura i+1).
            a = i * nth + j
            b = i * nth + (j + 1) % nth
            c = (i + 1) * nth + j
            d = (i + 1) * nth + (j + 1) % nth
            F.append([a, b, d])
            F.append([a, d, c])
    return V, np.array(F)


def save_ply(path, V, F):
    """Grava a malha (vértices V em cm e triângulos F) num ficheiro PLY de texto, para o MeshLab."""
    with open(path, "w") as f:
        # Cabeçalho: número de vértices e de faces; cada face é uma lista de índices.
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(V)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
        for v in V:
            f.write(f"{v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        # "3" = número de vértices da face (triângulo).
        for t in F:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def main():
    """Programa principal: lê um ou mais logs, junta os pontos, constrói a banda e grava
    a malha (.ply), as figuras (.png) e a vista interativa (.html, se houver plotly)."""
    ap = argparse.ArgumentParser(description="Surface (side band) from scan point clouds")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--dtheta", type=float, default=DTHETA)
    ap.add_argument("--dz", type=float, default=DZ)
    args = ap.parse_args()

    # --- ler cada log e centrar os seus pontos no eixo do objeto ---
    pts, sensors = [], []
    for f in args.logs:
        P, R, H = load_points(f)
        cx, cy = circle_centre(P[:, 0], P[:, 1])    # cada colocação tem o seu próprio centro
        # Desloca-se a origem para o eixo do objeto.
        P[:, 0] -= cx
        P[:, 1] -= cy
        pts.append(P)
        sensors.append((R, H))
        print(f"{os.path.basename(f)}: {len(P)} points, centre offset {np.hypot(cx, cy):.2f} cm")
    # Coordenadas cilíndricas em relação ao eixo do objeto: ângulo (rad), raio (cm), altura (cm).
    P = np.vstack(pts)
    th = np.arctan2(P[:, 1], P[:, 0])
    r = np.hypot(P[:, 0], P[:, 1])
    z = P[:, 2]

    th_c, z_c, rad, measured, count = build_band(th, z, r, args.dtheta, args.dz)
    V, F = band_mesh(th_c, z_c, rad)
    # Máscara medido/preenchido por vértice (mesma ordem de V) e fração de nós medidos.
    meas_v = measured.ravel()
    frac = 100 * measured.mean()
    print(f"Surface: {len(th_c)} x {len(z_c)} nodes ({args.dtheta} deg x {args.dz} cm), "
          f"z from {z_c[0]:.1f} to {z_c[-1]:.1f} cm, {frac:.0f}% measured, "
          f"{100 - frac:.0f}% filled")

    base = os.path.splitext(args.logs[0])[0] + "_malha"
    save_ply(base + ".ply", V, F)

    # ---- figura estática: duas vistas 3D + mapa medido/preenchido ----
    fig = plt.figure(figsize=(16, 5.5))
    # Repete-se a primeira coluna no fim para a superfície desenhada fechar a volta (sem fenda).
    TH, Z = np.meshgrid(np.r_[th_c, th_c[:1]], z_c)
    RR = np.c_[rad, rad[:, :1]]
    X, Y = RR * np.cos(TH), RR * np.sin(TH)
    M = np.c_[measured, measured[:, :1]]
    # Cor RGBA por nó: azul = medido, cinzento = preenchido.
    colors = np.where(M[..., None], np.array([0.23, 0.55, 0.76, 1.0]), np.array([0.78, 0.78, 0.78, 1.0]))
    lim = np.abs(np.r_[X.ravel(), Y.ravel()]).max() + 0.5
    # Duas vistas 3D: (elevação, azimute) em graus.
    for k, (el, az) in enumerate([(25, -60), (5, 30)]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.plot_surface(X, Y, Z, facecolors=colors,
                        rstride=1, cstride=1, linewidth=0.15, edgecolor=(0, 0, 0, 0.15),
                        shade=True, antialiased=True)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=2, c="k", alpha=0.25)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, z_c[-1] + 1)
        # Mesma escala nos três eixos.
        ax.set_box_aspect((2 * lim, 2 * lim, z_c[-1] + 1))
        ax.view_init(el, az)
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)"); ax.set_zlabel("z (cm)")
        ax.set_title("Side surface (blue = measured, grey = filled)" if k == 0 else "Side view")
    # Terceiro painel: a banda "desenrolada" (ângulo na horizontal, altura na vertical), cor = r.
    # Os nós preenchidos ficam NaN, ou seja, em branco.
    ax3 = fig.add_subplot(1, 3, 3)
    im = ax3.imshow(np.where(measured, rad, np.nan), origin="lower", aspect="auto", cmap="viridis",
                    extent=[-180, 180, z_c[0] - args.dz / 2, z_c[-1] + args.dz / 2])
    ax3.set_xlabel("angle around the axis (deg)"); ax3.set_ylabel("z (cm)")
    ax3.set_title("Distance from the axis, unrolled (white = filled)")
    fig.colorbar(im, ax=ax3, label="distance from axis (cm)")
    plt.tight_layout()
    plt.savefig(base + ".png", dpi=130)
    plt.close(fig)

    # ---- vista interativa (só se o plotly estiver instalado) ----
    try:
        import plotly.graph_objects as go
        # (variável não usada adiante: a cor da malha usa diretamente meas_v)
        intensity = np.where(meas_v, V[:, 2], np.nan)
        fig = go.Figure()
        # Malha com cor por vértice: intensidade 1 = medido (azul), 0 = preenchido (cinzento).
        fig.add_mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                       intensity=np.where(meas_v, 1.0, 0.0), colorscale=[[0, "#b8b8b8"], [1, "#3b8bc2"]],
                       showscale=False, flatshading=False, opacity=1.0, name="surface",
                       lighting=dict(ambient=0.45, diffuse=0.8, specular=0.2, roughness=0.6),
                       hovertemplate="x %{x:.1f}, y %{y:.1f}, z %{z:.1f} cm<extra></extra>")
        # Pontos escondidos por defeito (clicar na legenda para os mostrar).
        fig.add_scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name="points",
                          marker=dict(size=2, color="black", opacity=0.5), visible="legendonly")
        fig.update_layout(title=f"{os.path.basename(args.logs[0])}: side surface "
                                f"(blue = measured, grey = filled)",
                          scene=dict(aspectmode="data", xaxis_title="x (cm)", yaxis_title="y (cm)",
                                     zaxis_title="z (cm)"),
                          margin=dict(l=0, r=0, t=40, b=0))
        fig.write_html(base + ".html", include_plotlyjs=True)
        print(f"Saved {base}.html, .png, .ply")
    except ImportError:
        print(f"Saved {base}.png, .ply (install plotly for the interactive view)")


if __name__ == "__main__":
    main()
