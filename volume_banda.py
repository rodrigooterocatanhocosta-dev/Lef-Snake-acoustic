"""volume_banda.py - volume da buzina a partir da superfície lateral (banda do malha3d), com o viés
radial da banda calibrado, para cada distância do sensor R, na lata (raio conhecido 3,00 cm).

Utilização (na pasta com os logs e com malha3d.py e scan3d.py):
    python volume_banda.py
Não tem argumentos: os ficheiros estão escritos no código (CAN e f"buzina{R}.txt").
ATENÇÃO: todo o cálculo está ao nível do módulo, por isso também corre quando o ficheiro é
importado (p.ex. pelo validar.py), e precisa dos logs reais da lata e da buzina na pasta.

Fórmula:
    V = integral_0^ztop  1/2 * soma_theta (r - b(R))^2 dtheta dz
ou seja, em cada altura a área da secção em coordenadas polares, A(z) = 1/2 * integral r(theta)^2
dtheta, com o raio corrigido do viés, e depois o integral de A(z) em altura (regra dos trapézios).
As linhas abaixo/acima da banda são prolongadas com a primeira/última linha (A constante).

Método, passo a passo:
  1. Calibração do viés: para cada R, faz-se a banda da lata (mesmo método, mesmo R) e
     b(R) = raio médio quadrático da banda - 3,00 cm. O viés vem sobretudo da largura do feixe
     (o ponto mais próximo dentro do cone é posto no eixo do feixe -> pontos deslocados para fora)
     e depende de R, porque a largura do cone no objeto cresce com a distância.
  2. Banda da buzina ao mesmo R, subtração de b(R) a todos os raios, área por linha, integral
     em z de 0 até ao topo ZTOP.
  3. Incerteza, por propagação "um de cada vez" (diferenças finitas simétricas):
       dVb = [V(b + sigma_b) - V(b - sigma_b)] / 2,   dVz = [V(ztop + sigma_z) - V(ztop - sigma_z)] / 2,
       sigma_V = sqrt(dVb^2 + dVz^2)   (em quadratura: as duas fontes são independentes).
     sigma_b é a incerteza com que o próprio scanner mede o raio da lata (sigma_D/2 do
     dimensoes.py, com beta fixo): é a precisão com que se conhece o viés.
  4. Teste z em relação ao volume exterior de referência da buzina, 126 +- 1 cm^3 (= ml, valor
     medido independentemente pelo grupo):
       z = (V - 126) / sqrt(sigma_V^2 + 1^2).

Saídas: resumo no ecrã e o ficheiro buzina_volume.json (viés por R, volume, incerteza, volume
sem correção "raw", fração de nós medidos e z para cada R).

Hipóteses e limitações conhecidas:
  - log.txt é ASSUMIDO como sendo a lata a R = 10 (é o único varrimento a R = 10).
  - A altura da buzina ZTOP = 15,0 +- 0,5 cm é ASSUMIDA (o topo só é visto a R = 14).
  - O viés calibrado na lata (raio 3 cm, parede vertical) não é exatamente o viés de um objeto
    mais fino ou de forma diferente: o validar.py mostra, em dados simulados, que o método
    sobrestima o volume de objetos finos, sobretudo de perto (R = 8-10 cm: ~160-200 ml para um
    cilindro de 127 ml; R = 12-14: ~130-150 ml). Os resultados a R = 12-14 são os mais fiáveis.
  - A banda é r(theta, z): só descreve secções "em estrela" em relação ao eixo.
"""
import numpy as np, json, malha3d as mh
# Logs da lata usados na calibração, por distância R (cm) do sensor.
CAN = {8: ["lata3.txt"], 10: ["log.txt"],   # log.txt ASSUMIDO como a lata (único varrimento a R = 10)
       12: ["lata1.txt", "lata2.txt"], 14: ["lata4.txt"]}
# Incerteza do viés, sigma_b (cm), por R: sigma_D/2 do dimensoes.py (beta fixo).
CAN_SIG_R = {8: 0.16, 10: 0.18, 12: 0.18, 14: 0.17}
def band(f, maxr=7.0):
    """Banda (superfície lateral) de um log, centrada no eixo do objeto.

    Entradas: f - ficheiro de log; maxr - raio máximo (cm) para aceitar pontos (resto = fundo).
    Saída: (P, thc, zc, rad, meas)
        P - pontos (N, 7) centrados no eixo do objeto (cm);
        thc (rad), zc (cm) - ângulos e alturas dos nós da grelha;
        rad (nz, nth) - raio da banda em cada nó (cm), AINDA com o viés;
        meas (nz, nth) - True nos nós medidos (False = preenchidos).
    Usa exatamente o mesmo procedimento do malha3d.py (load_points, circle_centre, build_band,
    com DTHETA e DZ do malha3d), para que a lata e a buzina tenham o mesmo viés.
    """
    # Altera a configuração global do malha3d (MAX_RADIUS) antes de ler os pontos.
    mh.MAX_RADIUS = maxr
    # Ler pontos e centrar no eixo do objeto (ajuste de circunferência visto de cima).
    P, R, H = mh.load_points(f); cx, cy = mh.circle_centre(P[:, 0], P[:, 1])
    P = P.copy(); P[:, 0] -= cx; P[:, 1] -= cy
    # Coordenadas cilíndricas: ângulo (rad) e distância ao eixo (cm).
    th = np.arctan2(P[:, 1], P[:, 0]); r = np.hypot(P[:, 0], P[:, 1])
    thc, zc, rad, meas, _ = mh.build_band(th, P[:, 2], r, mh.DTHETA, mh.DZ)
    return P, thc, zc, rad, meas
def volume(zc, rad, b, ztop):
    """Volume (cm^3) delimitado pela banda, depois de subtrair o viés radial b.

    Entradas:
        zc (nz,) - alturas das linhas da banda (cm);
        rad (nz, nth) - raio em cada nó (cm), nós igualmente espaçados em ângulo;
        b - viés radial a subtrair (cm);  ztop - altura do topo do objeto (cm).
    Saída: V = integral de 0 a ztop de A(z) dz, com A(z) = 1/2 * soma_j (r_j - b)^2 * dtheta.
    """
    # Passo angular: os nós cobrem 2*pi em partes iguais.
    dth = 2 * np.pi / rad.shape[1]
    # Área polar de cada linha: soma de setores 1/2 r^2 dtheta. O maximum(.., 0) impede que um
    # raio corrigido negativo (b maior do que r) conte como área positiva ao ser elevado ao quadrado.
    A = 0.5 * (np.maximum(rad - b, 0) ** 2).sum(1) * dth          # área de cada linha
    # Prolongar a banda até ao prato (z = 0) com a área da primeira linha e até ztop com a da
    # última: a banda só cobre os percentis 2-98 das alturas, e perto do prato e do topo há poucos
    # ecos úteis, por isso assume-se secção constante nesses troços.
    z = np.r_[0.0, zc, ztop]; A = np.r_[A[0], A, A[-1]]
    # Se a banda passar acima de ztop, corta-se aí ...
    m = (z <= ztop); z, A = z[m], A[m]
    # ... e fecha-se exatamente em ztop com a área da última linha que ficou.
    if z[-1] < ztop: z = np.r_[z, ztop]; A = np.r_[A, A[-1]]
    # Integral em altura pela regra dos trapézios (np.trapezoid exige numpy >= 2.0).
    return np.trapezoid(A, z)
# ---------- 1. calibração do viés radial b(R) na lata ----------
# Para cada R: raio médio quadrático (RMS) dos nós MEDIDOS da banda da lata menos o raio verdadeiro
# 3,00 cm. Usa-se o RMS (e não a média simples) porque a área depende de r^2: um círculo com o raio
# RMS tem a mesma área média que a banda, que é o que interessa para o volume. Com vários logs ao
# mesmo R faz-se a média dos viéses.
bias = {}
for R, fs in CAN.items():
    v = []
    for f in fs:
        P, thc, zc, rad, meas = band(f)
        v.append(np.sqrt(np.mean(rad[meas] ** 2)) - 3.0)
    bias[R] = (float(np.mean(v)), CAN_SIG_R[R])
print("radial bias b(R) from the can:", {k: (round(a, 2), s) for k, (a, s) in bias.items()})
# Altura assumida da buzina e a sua incerteza (cm); o topo só se vê no varrimento a R = 14.
ZTOP, SZTOP = 15.0, 0.5
# ---------- 2-4. volume da buzina para cada R, incerteza e teste z ----------
out = {}
for R in [8, 10, 12, 14]:
    f = f"buzina{R}.txt"
    P, thc, zc, rad, meas = band(f)
    # Viés calibrado na lata ao MESMO R e a sua incerteza.
    b, sb = bias[R]
    # Volume corrigido e, para comparação, sem correção do viés (b = 0).
    V = volume(zc, rad, b, ZTOP); Vraw = volume(zc, rad, 0.0, ZTOP)
    # Propagação um de cada vez: varia-se só b em +-sigma_b, depois só ztop em +-sigma_z.
    dVb = (volume(zc, rad, b + sb, ZTOP) - volume(zc, rad, b - sb, ZTOP)) / 2
    dVz = (volume(zc, rad, b, ZTOP + SZTOP) - volume(zc, rad, b, ZTOP - SZTOP)) / 2
    # Combinação em quadratura: sigma_V = sigma(b) (+) sigma(topo).
    sV = float(np.hypot(dVb, dVz))
    # Teste z contra o volume de referência 126 cm^3 (incerteza 1 cm^3); |z| < 2 = compatível.
    z = (V - 126) / np.hypot(sV, 1.0)
    out[R] = dict(V=float(V), sV=sV, Vraw=float(Vraw), b=b, sb=sb, frac=float(meas.mean()), z=float(z))
    print(f, f"V = {V:.0f} +- {sV:.0f} cm3 (raw {Vraw:.0f}), b = {b:+.2f}, measured {100*meas.mean():.0f}%, z = {z:+.2f}")
# Gravar tudo em JSON (lido depois pelo build_excel.py).
json.dump(dict(bias=bias, buzina=out, ztop=[ZTOP, SZTOP]), open("buzina_volume.json", "w"), indent=1)
