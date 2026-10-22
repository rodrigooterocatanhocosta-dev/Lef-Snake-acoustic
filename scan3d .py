"""
scan3d.py - converte o registo (log) de um varrimento ultrassónico num conjunto de pontos 3D e em figuras.

Utilização:
    python scan3d.py log.txt                 # último varrimento completo do ficheiro
    python scan3d.py log.txt --scan 2        # o 2.º varrimento do ficheiro (1 = primeiro)
    python scan3d.py log.txt --list          # lista os varrimentos encontrados no ficheiro
    python scan3d.py log.txt --R 6.5 --H 12  # substitui a geometria lida do log

O log pode ser colado diretamente do Serial Monitor do Arduino: linhas de comentário (#),
menus repetidos e varrimentos interrompidos são tratados automaticamente.

O que é medido
--------------
O objeto está num prato rodado por um motor de passo. O HC-SR04 está num braço inclinado por um
servo; o eixo do servo está à distância horizontal R do eixo do prato e à altura H acima do topo
do prato. Cada linha de dados do log é:
    posicao, angulo_prato (graus), inclinacao (graus, positiva = feixe para baixo), distancia (cm)
A distância já vem calculada pelo Arduino a partir do tempo de voo: d = v_som * t / 2
(ida e volta, por isso a divisão por 2). Leitura sem eco = valor <= 0.

Método, passo a passo
---------------------
  1. parse_log: lê o ficheiro, separa os varrimentos (começam numa linha "# posicao" e acabam em
     "# Fim"), guarda a geometria R, H da linha "# Geometria" e marca os interrompidos.
  2. Filtra as leituras sem eco (distância <= 0) e, se pedido, as mais longas que MAX_DIST.
  3. merge_repeats: inclinações consecutivas na mesma posição do prato que dão a mesma distância
     são o MESMO ponto do objeto visto por várias inclinações (o feixe tem ~12-13 graus de
     semi-abertura); juntam-se num só ponto, na inclinação média, com um contador de repetições.
  4. to_points: correção de escala/offset da distância, correção do triângulo emissor-recetor
     (separação de 2,6 cm), posição dos transdutores em relação ao eixo do servo, e o ponto é
     colocado SOBRE O EIXO DO FEIXE à distância medida. Depois roda-se para trás o ângulo do prato,
     para passar do referencial do laboratório para o referencial do objeto.
  5. Pontos abaixo de Z_MIN são tratados como ecos do prato e rejeitados.
  6. Gravação dos pontos e das figuras.

Saídas (ao lado do ficheiro de log, com o nome <log>_scanN_*):
    _pontos.csv / _pontos.npy   matriz de pontos (colunas abaixo)
    _pontos.ply                 nuvem de pontos para o MeshLab
    _figura.png                 vista 3D, vista de cima e perfil lateral
    _figura.html                vista 3D interativa (só se o plotly estiver instalado)

Colunas da matriz de pontos:
    x, y, z (cm, referencial do objeto)   angulo_prato (graus)   inclinacao (graus)
    distancia (cm, corrigida)             repeticoes (leituras juntas nesse ponto)

Limitações conhecidas
---------------------
  - O ponto é posto no eixo do feixe, mas o HC-SR04 devolve o eco do ponto MAIS PRÓXIMO que
    responda dentro do cone. Para objetos convexos isto empurra os pontos para fora (viés radial
    positivo, ~1 cm). Este script NÃO corrige isso: as dimensões com incerteza vêm dos ajustes do
    modelo do feixe (dimensoes.py, prisma.py, hemi.py), que modelam o cone explicitamente.
  - A resolução em distância é ~0,33 cm (passos visíveis nos logs).
  - A velocidade do som depende da temperatura (~0,17 % por grau C); um erro aí é um erro de escala
    e corrige-se com DIST_SCALE.
"""

import argparse
import os
import re

import numpy as np
import matplotlib
# Backend "Agg": desenha para ficheiro, sem precisar de janela (funciona em qualquer máquina).
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =================== CONFIGURAÇÕES (alterar aqui) ===================
# Geometria. None = ler R e H da linha "# Geometria" do log (é o normal; só se muda se o log
# não tiver essa linha ou se a geometria escrita no Arduino estiver errada).
R_SENSOR = None        # cm: distância HORIZONTAL do eixo do prato ao eixo do servo
H_SENSOR = None        # cm: distância VERTICAL do topo do prato ao eixo do servo

# Onde estão os transdutores em relação ao eixo do servo (medir com régua).
# Importa porque o servo roda o sensor em torno do SEU eixo; se os transdutores não estiverem
# sobre esse eixo, a origem do feixe desloca-se quando a inclinação muda.
PIVOT_FRONT = 0.0      # cm ao longo do feixe (positivo = no sentido do objeto)
PIVOT_UP = 0.0         # cm perpendicular ao feixe (positivo = para cima quando o feixe está horizontal)

# Correções da distância. Deixar DIST_OFFSET = 0 se o offset já for aplicado no Arduino.
DIST_OFFSET = 0.0      # cm, (real - medido); vem da calibração com distâncias conhecidas
DIST_SCALE = 1.0       # real = DIST_SCALE * medido + DIST_OFFSET; corrige p.ex. a velocidade do som
TR_SEPARATION = 2.6    # cm entre o centro do emissor e o do recetor (correção do triângulo)

# Desvio do centro do prato, se o prato estiver descentrado no motor (cm). Subtrai-se no fim,
# já no referencial do objeto.
PLATE_CX = 0.0
PLATE_CY = 0.0

# Filtragem
Z_MIN = 1.0            # cm: pontos mais baixos do que isto são ecos do prato (rejeitados)
MAX_DIST = None        # cm: ignorar leituras mais longas do que isto (None = sem limite);
                       # útil para cortar ecos da parede/fundo da sala

# Juntar inclinações consecutivas, na mesma posição do prato, que dão a mesma distância
MERGE = True           # False = cada leitura é um ponto (com repetições = 1)
MERGE_TOL = 0.35       # cm; ~ um passo de resolução (0,33 cm): diferenças menores são "a mesma"
# ===================================================================


def parse_log(path):
    """Lê um ficheiro de log e devolve a lista de varrimentos encontrados.

    Entrada:
        path - caminho do ficheiro de texto (pode ter lixo do Serial Monitor).
    Saída:
        lista de dicionários, um por varrimento, com:
            rows       - np.array (N, 4): posicao, angulo_prato (graus), inclinacao (graus),
                         distancia (cm), exatamente como vêm do Arduino;
            complete   - True se o varrimento acabou com a linha "# Fim";
            R, H       - geometria (cm) da última linha "# Geometria" vista ANTES do varrimento
                         (None se não houver);
            start_line - número da linha onde o varrimento começa (para o --list).

    Ideia: uma máquina de estados simples. Uma linha "# posicao" (cabeçalho das colunas) abre um
    varrimento novo; linhas com 4 números separados por vírgulas são dados; "# Fim" fecha-o como
    completo; qualquer outro texto que apareça depois de já haver dados significa que o
    varrimento foi interrompido (p.ex. reset do Arduino, menu impresso de novo) e é fechado como
    incompleto.
    """
    scans = []
    R = H = None
    current = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            # Linha de geometria, p.ex. "# Geometria: R=12.0 cm, H=12.0 cm". Fica guardada e é
            # atribuída aos varrimentos que começarem a seguir.
            m = re.search(r"Geometria:\s*R=([\d.]+)\s*cm,\s*H=([\d.]+)", line)
            if m:
                R, H = float(m.group(1)), float(m.group(2))
            # Cabeçalho das colunas = início de um varrimento novo (aceita com e sem acento).
            # Se havia um varrimento aberto com dados, fica guardado (sem "# Fim" = incompleto).
            if line.startswith("# posicao") or line.startswith("# posição"):
                if current and current["rows"]:
                    scans.append(current)
                current = dict(rows=[], complete=False, R=R, H=H, start_line=n)
                continue
            # Ainda não começou nenhum varrimento: ignorar (menus, configuração, etc.).
            if current is None:
                continue
            # Fim normal do varrimento.
            if line.startswith("# Fim"):
                current["complete"] = True
                scans.append(current)
                current = None
                continue
            # Linha de dados: exatamente 4 campos numéricos.
            parts = line.split(",")
            if len(parts) == 4:
                try:
                    current["rows"].append([float(p) for p in parts])
                    continue
                except ValueError:
                    pass
            # Qualquer outro texto depois de os dados terem começado = varrimento interrompido.
            # (Texto antes do primeiro dado, p.ex. linhas em branco, é ignorado.)
            if current["rows"] and line:
                scans.append(current)
                current = None
    # Ficheiro acabou a meio de um varrimento: guarda-se o que houver (incompleto).
    if current and current["rows"]:
        scans.append(current)
    # Listas -> arrays numpy para o resto do código.
    for s in scans:
        s["rows"] = np.array(s["rows"])
    return scans


def merge_repeats(d):
    """Junta inclinações consecutivas (mesma posição do prato) que deram a mesma distância.

    Entrada:
        d - array (N, 4): posicao, angulo_prato (graus), inclinacao (graus), distancia (cm),
            já sem leituras sem eco.
    Saída:
        array (M, 5), M <= N: posicao, angulo_prato, inclinacao MÉDIA do grupo,
        distancia MÉDIA do grupo, número de leituras juntas (repetições).

    Porquê: o feixe do HC-SR04 é largo (semi-abertura ~12-13 graus) e o sensor devolve o eco do
    ponto mais próximo dentro do cone. Quando o servo muda a inclinação em passos de 5 graus, o
    mesmo ponto do objeto (p.ex. uma aresta ou o ponto mais saliente) continua a ser o mais
    próximo durante várias inclinações e a distância não muda. Se cada leitura fosse posta no
    eixo do feixe, esse único ponto apareceria como uma "coluna" vertical de pontos falsos.
    Juntando-as, fica um só ponto na inclinação média (o melhor palpite para a direção real) e o
    número de repetições indica que aquele ponto é "forte".

    Critério: dentro de cada posição do prato, ordenam-se as leituras pela inclinação e
    percorrem-se em sequência; uma leitura entra no grupo atual se a sua distância diferir menos
    de MERGE_TOL da ÚLTIMA leitura do grupo e se for a inclinação seguinte (diferença <= 5 graus,
    com 0,01 de folga para arredondamentos). Caso contrário começa um grupo novo.
    """
    out = []
    # Uma posição do prato de cada vez (coluna 0 = índice da posição).
    for k in np.unique(d[:, 0]):
        s = d[d[:, 0] == k]
        # Ordenar por inclinação: o servo pode varrer nos dois sentidos (ida e volta).
        s = s[np.argsort(s[:, 2])]
        group = [s[0]]
        for row in s[1:]:
            same = abs(row[3] - group[-1][3]) < MERGE_TOL
            next_tilt = row[2] - group[-1][2] <= 5.01
            if same and next_tilt:
                group.append(row)
            else:
                out.append(group)
                group = [row]
        out.append(group)
    # Cada grupo -> uma linha: posição e ângulo do prato do primeiro, médias de inclinação e
    # distância, e o tamanho do grupo.
    merged = [[g[0][0], g[0][1], np.mean([r[2] for r in g]),
               np.mean([r[3] for r in g]), len(g)] for g in out]
    return np.array(merged)


def to_points(d, R, H):
    """Converte (ângulo do prato, inclinação, distância) em pontos x, y, z no referencial do objeto.

    Entradas:
        d - array (N, 5), colunas: posicao, angulo_prato (graus), inclinacao (graus),
            distancia (cm, tal como medida), repeticoes;
        R - distância horizontal eixo do prato -> eixo do servo (cm);
        H - altura do eixo do servo acima do topo do prato (cm).
    Saída:
        array (N, 7): x, y, z (cm, referencial do objeto, origem no eixo do prato ao nível do
        topo do prato, z para cima), angulo_prato (graus), inclinacao (graus),
        distancia corrigida (cm), repeticoes.

    Ideia: no referencial do LABORATÓRIO o sensor está sempre no plano x-z, em (R, 0, H), a
    apontar para o eixo do prato (sentido -x) e inclinado para baixo de 'inc'. O ponto é posto
    sobre o eixo do feixe à distância medida. Como é o objeto que roda (com o prato) e não o
    sensor, roda-se o ponto de volta pelo ângulo do prato para o escrever no referencial que
    roda com o objeto; assim todas as posições do prato ficam no mesmo referencial.
    """
    ang = np.radians(d[:, 1])
    inc = np.radians(d[:, 2])
    # Correção linear da distância (calibração): real = escala * medido + offset.
    dist = DIST_SCALE * d[:, 3] + DIST_OFFSET
    # Correção do triângulo: o emissor e o recetor estão lado a lado, separados de s = 2,6 cm.
    # O som vai do emissor ao ponto e volta ao recetor; o Arduino dá metade desse caminho,
    # que para um ponto no eixo (à distância d do ponto médio entre os dois) é
    # sqrt(d^2 + (s/2)^2). Invertendo: d = sqrt(medido^2 - (s/2)^2). O maximum(.., 0) evita a
    # raiz de um número negativo em leituras absurdamente curtas. O efeito só conta de perto
    # (a 5 cm são ~0,17 cm; a 15 cm ~0,06 cm).
    dist = np.sqrt(np.maximum(dist**2 - (TR_SEPARATION / 2) ** 2, 0))

    # Direção do feixe u (para o eixo do prato, -x, e para baixo, -z, quando inc > 0) e a sua
    # normal n "para cima", ambas vetores unitários no plano x-z. (u e n são perpendiculares.)
    ux, uz = -np.cos(inc), -np.sin(inc)
    nx, nz = -np.sin(inc), np.cos(inc)
    # Origem do feixe: eixo do servo (R, H) mais o desvio dos transdutores, expresso nos eixos que
    # rodam com o sensor (PIVOT_FRONT ao longo de u, PIVOT_UP ao longo de n). Com os dois a zero
    # a origem é simplesmente (R, H).
    sx = R + PIVOT_FRONT * ux + PIVOT_UP * nx
    sz = H + PIVOT_FRONT * uz + PIVOT_UP * nz

    # Ponto sobre o eixo do feixe, à distância corrigida (no referencial do laboratório, y = 0).
    x = sx + dist * ux
    z = sz + dist * uz
    # O prato rodou de 'ang': rodar o ponto de volta para o referencial do objeto. O ponto do
    # laboratório (x, 0) corresponde no objeto a (x cos(ang), -x sin(ang)), isto é, ao ângulo
    # -ang. O sinal segue o sentido de rotação do motor; se estivesse trocado, o objeto
    # apareceria espelhado mas com as mesmas dimensões. A altura z não muda com a rotação.
    # Por fim desconta-se o descentramento do prato (PLATE_CX, PLATE_CY).
    xo = x * np.cos(ang) - PLATE_CX
    yo = -x * np.sin(ang) - PLATE_CY
    return np.c_[xo, yo, z, d[:, 1], d[:, 2], dist, d[:, 4]]


def save_ply(path, P):
    """Grava os pontos num ficheiro PLY de texto (formato de nuvem de pontos do MeshLab).

    Entradas: path - nome do ficheiro; P - array (N, >=3) com x, y, z em cm nas 3 primeiras
    colunas. Saída: nenhuma (escreve o ficheiro). Só se gravam as coordenadas, com 3 casas
    decimais (0,01 mm, muito abaixo da resolução do sensor).
    """
    with open(path, "w") as f:
        # Cabeçalho PLY: número de vértices e as três propriedades de cada um.
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(P)}\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in P:
            f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")


def make_figure(path, P, rejected, R, H, title):
    """Desenha e grava a figura estática (PNG) com três painéis.

    Entradas:
        path     - ficheiro de saída (.png);
        P        - pontos aceites, array (N, 7) devolvido por to_points;
        rejected - pontos rejeitados como ecos do prato (mesmo formato), desenhados a cinzento;
        R, H     - geometria do sensor (cm), para desenhar a sua trajetória e posição;
        title    - título geral da figura.
    Painéis:
        1. vista 3D, cor = inclinação do feixe; inclui a circunferência que o sensor descreve
           em relação ao objeto (raio R, altura H) e o eixo do prato;
        2. vista de cima (x, y); o tamanho do marcador cresce com o número de repetições;
        3. perfil lateral: distância ao eixo do prato sqrt(x^2+y^2) contra z, juntando todas as
           rotações (para um objeto de revolução, isto é o seu perfil).
    """
    fig = plt.figure(figsize=(18, 5.5))
    fig.subplots_adjust(wspace=0.55)
    # Ângulos para desenhar a circunferência do sensor.
    th = np.linspace(0, 2 * np.pi, 120)
    # Limites dos eixos: cobrir o objeto e a posição do sensor, com 1 cm de margem.
    lim = max(R, np.abs(P[:, :2]).max() if len(P) else 1) + 1
    zmax = max(H, P[:, 2].max() if len(P) else 1) + 1

    # --- painel 1: vista 3D ---
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=8, c=P[:, 4], cmap="viridis", depthshade=False)
    if len(rejected):
        ax.scatter(rejected[:, 0], rejected[:, 1], rejected[:, 2], s=6, c="#aaaaaa",
                   depthshade=False, label="rejected (plate)")
    ax.plot(R * np.cos(th), R * np.sin(th), np.full(th.size, H), ":", c="gray", label="sensor path")
    ax.plot([0, 0], [0, 0], [0, zmax], "k--", lw=1, label="plate axis")
    ax.scatter([R], [0], [H], marker="s", s=50, c="k", label="sensor")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, zmax)
    # Escala igual nos três eixos, para não deformar o objeto.
    ax.set_box_aspect((2 * lim, 2 * lim, zmax))
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)"); ax.set_zlabel("z (cm)")
    ax.set_title("3D"); ax.legend(fontsize=7, loc="upper left")

    # --- painel 2: vista de cima; marcador maior = mais leituras juntas (coluna 6) ---
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(P[:, 0], P[:, 1], s=8 + 6 * (P[:, 6] - 1), c=P[:, 4], cmap="viridis")
    ax2.plot(0, 0, "k+", ms=12)
    ax2.set_aspect("equal"); ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("x (cm)"); ax2.set_ylabel("y (cm)"); ax2.set_title("Top view")

    # --- painel 3: perfil lateral (raio ao eixo do prato vs altura, todas as rotações) ---
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(np.hypot(P[:, 0], P[:, 1]), P[:, 2], s=8 + 6 * (P[:, 6] - 1), c=P[:, 4], cmap="viridis")
    ax3.plot(R, H, "ks", ms=8, label="sensor")
    ax3.set_xlim(0, lim); ax3.set_ylim(0, zmax); ax3.set_aspect("equal")
    ax3.set_xlabel("distance from plate axis (cm)"); ax3.set_ylabel("z (cm)")
    ax3.set_title("Side profile (all rotations)"); ax3.legend(fontsize=7)

    # Uma só barra de cor (inclinação) partilhada pelos três painéis.
    fig.colorbar(sc, ax=[ax, ax2, ax3], shrink=0.7, label="inclination (deg)")
    fig.suptitle(title)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_html(path, P, rejected, R, H, title):
    """Grava uma vista 3D interativa (HTML, plotly) dos pontos.

    Entradas: as mesmas de make_figure (path deve acabar em .html).
    Saída: True se o ficheiro foi gravado; False se o plotly não estiver instalado (nesse caso
    não se faz nada; o resto do script funciona na mesma).
    Ao passar o rato sobre um ponto vê-se o ângulo do prato, a inclinação, a distância corrigida
    e o número de leituras juntas ("x3" = 3 leituras).
    """
    # O plotly é opcional: importa-se só aqui.
    try:
        import plotly.graph_objects as go
    except ImportError:
        return False
    th = np.linspace(0, 2 * np.pi, 120)
    fig = go.Figure()
    # Pontos, cor = inclinação; texto do "hover" a partir das colunas 3 a 6.
    fig.add_scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name="points",
        marker=dict(size=3, color=P[:, 4], colorscale="Viridis", colorbar=dict(title="inc (deg)")),
        text=[f"plate {a:.0f}°, inc {i:.0f}°, d={r:.2f} cm, x{int(n)}" for a, i, r, n in P[:, 3:7]],
        hoverinfo="text")
    if len(rejected):
        fig.add_scatter3d(x=rejected[:, 0], y=rejected[:, 1], z=rejected[:, 2], mode="markers",
                          name="rejected (plate)", marker=dict(size=2, color="#aaaaaa"))
    # Trajetória do sensor e eixo do prato, como referência.
    fig.add_scatter3d(x=R * np.cos(th), y=R * np.sin(th), z=np.full(th.size, H), mode="lines",
                      name="sensor path", line=dict(color="gray", dash="dot"))
    fig.add_scatter3d(x=[0, 0], y=[0, 0], z=[0, max(H, 1) + 1], mode="lines", name="plate axis",
                      line=dict(color="black", dash="dash"))
    # aspectmode="data": a mesma escala nos três eixos.
    fig.update_layout(title=title, scene=dict(aspectmode="data", xaxis_title="x (cm)",
                      yaxis_title="y (cm)", zaxis_title="z (cm)"), margin=dict(l=0, r=0, t=40, b=0))
    # include_plotlyjs=True: o HTML fica autónomo (abre sem internet).
    fig.write_html(path, include_plotlyjs=True)
    return True


def main():
    """Programa principal: lê os argumentos, escolhe o varrimento, calcula os pontos e grava tudo.

    Saída: devolve o array P de pontos aceites (útil se main() for chamada a partir de outro
    script ou de uma consola Python); também imprime um resumo e grava os ficheiros descritos
    no início do módulo.
    """
    # --- argumentos da linha de comandos ---
    ap = argparse.ArgumentParser(description="Ultrasonic scan log -> 3D points and figures")
    ap.add_argument("log")
    ap.add_argument("--scan", type=int, help="which scan in the file (1 = first); default = last complete")
    ap.add_argument("--list", action="store_true", help="list the scans in the file and exit")
    ap.add_argument("--R", type=float, help="override R (cm)")
    ap.add_argument("--H", type=float, help="override H (cm)")
    args = ap.parse_args()

    scans = parse_log(args.log)
    if not scans:
        raise SystemExit("No scan data found in the file.")

    # --list: só mostra o que há no ficheiro e sai.
    if args.list:
        for i, s in enumerate(scans, 1):
            print(f"{i}: line {s['start_line']}, {len(s['rows'])} readings, "
                  f"{'complete' if s['complete'] else 'INTERRUPTED'}, R={s['R']} H={s['H']}")
        return

    # Escolha do varrimento: o pedido com --scan (contagem a partir de 1); senão o último
    # completo; se nenhum estiver completo, o último que houver.
    if args.scan:
        idx = args.scan - 1
    else:
        complete = [i for i, s in enumerate(scans) if s["complete"]]
        idx = complete[-1] if complete else len(scans) - 1
    scan = scans[idx]

    # Geometria, por ordem de prioridade: linha de comandos > SETTINGS > log.
    # (Nota: com "or", um valor 0 conta como "não dado".)
    R = args.R or R_SENSOR or scan["R"]
    H = args.H or H_SENSOR or scan["H"]
    if R is None or H is None:
        raise SystemExit("R/H not found in the log: set R_SENSOR/H_SENSOR or use --R/--H.")

    # Filtro das leituras: só as que tiveram eco (distância > 0) e, se definido, até MAX_DIST.
    d = scan["rows"]
    valid = d[:, 3] > 0
    if MAX_DIST is not None:
        valid &= d[:, 3] <= MAX_DIST
    d = d[valid]
    # Juntar repetições; se MERGE = False, acrescenta-se a coluna de repetições = 1 para que o
    # formato seja o mesmo (5 colunas) que to_points espera.
    d = merge_repeats(d) if MERGE else np.c_[d, np.ones(len(d))]

    # Pontos no referencial do objeto; abaixo de Z_MIN são ecos do prato (guardados à parte só
    # para aparecerem a cinzento nas figuras).
    P_all = to_points(d, R, H)
    keep = P_all[:, 2] >= Z_MIN
    P, rejected = P_all[keep], P_all[~keep]

    # --- gravação dos pontos (CSV legível, NPY para outros scripts, PLY para o MeshLab) ---
    base = os.path.splitext(args.log)[0] + f"_scan{idx + 1}"
    header = "x_cm,y_cm,z_cm,plate_angle_deg,inclination_deg,distance_cm,repeats"
    np.savetxt(base + "_pontos.csv", P, delimiter=",", header=header, comments="", fmt="%.3f")
    np.save(base + "_pontos.npy", P)
    save_ply(base + "_pontos.ply", P)

    # --- figuras ---
    title = f"Scan {idx + 1}: R={R} cm, H={H} cm, {len(P)} points"
    make_figure(base + "_figura.png", P, rejected, R, H, title)
    html = make_html(base + "_figura.html", P, rejected, R, H, title)

    # --- resumo no ecrã ---
    # Distância de cada ponto ao eixo do prato. Para um cilindro centrado, a mediana é uma
    # primeira estimativa (enviesada para fora, ver limitações) do raio; o intervalo 10-90 %
    # mostra a dispersão sem ser dominado por ecos isolados.
    r = np.hypot(P[:, 0], P[:, 1])
    print(f"Scan {idx + 1} of {len(scans)} ({'complete' if scan['complete'] else 'interrupted'}), R={R}, H={H}")
    print(f"Readings with echo: {int(valid.sum())} of {len(scan['rows'])}")
    print(f"Points kept: {len(P)} (rejected as plate: {len(rejected)})"
          + (f", merged from {int(P[:, 6].sum())} readings" if MERGE else ""))
    if len(P):
        print(f"Radius from axis: median {np.median(r):.2f} cm (10-90%: {np.percentile(r, 10):.2f}-{np.percentile(r, 90):.2f})")
        print(f"Height range:     {P[:, 2].min():.2f} - {P[:, 2].max():.2f} cm")
    print("Saved:", base + "_pontos.csv/.npy/.ply,", base + "_figura.png" + (", .html" if html else ""))
    return P


if __name__ == "__main__":
    main()
