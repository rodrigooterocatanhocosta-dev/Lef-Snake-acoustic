#include <Servo.h>

// ================= CONFIGURAÇÃO =================
// --- Pinos ---
const int PINO_TRIG = 2, PINO_ECHO = 4;
const int PINOS_STEPPER[4] = {13, 12, 11, 10};  // IN1..IN4 (se só vibrar: {4, 6, 5, 7})
const int PINO_SERVO = 6;

// --- Geometria (cm) - medir no setup real ---
float R_SENSOR = 14.0;      // horizontal: eixo do prato -> eixo do servo
float H_SENSOR = 12.0;      // altura do sensor acima do prato
float MARGEM_PRATO = 0;   // ignora ecos até 1 cm acima do prato

// --- Prato (stepper 28BYJ-48) ---
const int PASSOS_VOLTA = 2048;
int nPosicoes = 72;         // 72 = de 5 em 5 graus
float rpm = 10;             // máx. ~15
int esperaPratoMs = 200;

// --- Inclinação (graus abaixo da horizontal) ---
bool usarInclinacao = true;
int incMin = 20, incMax = 60, incPasso = 5;
int incFixa = 45;
int servoOffset = 55;        // ajuste do alinhamento a 45 graus
bool servoInvertido = false;
const int SERVO_MIN_US = 500, SERVO_MAX_US = 2400;
int servoEsperaMs = 300;

// --- Medição ---
int nLeituras = 10;          // mediana de N (máx. 15)
float distMin = 3.0;
float distMaxManual = 30.0;
bool janelaAuto = true;     // corta o eco da placa conforme a inclinação
float temperatura = 19.5;
float distOffset = 0.5;     // cm: distância real - distância medida (calibração com a placa)
// =================================================

const int MAX_LEITURAS = 15;
const uint8_t SEQ[4][4] = {{1,1,0,0}, {0,1,1,0}, {0,0,1,1}, {1,0,0,1}};
Servo servo;
int fase = 0;
int incAtual = 45;

// ---------- Stepper ----------
void desligarBobinas() {
  for (int i = 0; i < 4; i++) digitalWrite(PINOS_STEPPER[i], LOW);
}

void rodarPassos(long n) {
  int dir = (n >= 0) ? 1 : -1;
  unsigned long intervalo = (unsigned long)(60000000.0 / (rpm * PASSOS_VOLTA));
  for (long i = 0; i < labs(n); i++) {
    fase = (fase + dir + 4) % 4;
    for (int j = 0; j < 4; j++) digitalWrite(PINOS_STEPPER[j], SEQ[fase][j]);
    delayMicroseconds(intervalo);
  }
  desligarBobinas();
}

// ---------- Servo ----------
void moverServo(int inc) {
  int a = inc + servoOffset;
  if (servoInvertido) a = 180 - a;
  servo.attach(PINO_SERVO, SERVO_MIN_US, SERVO_MAX_US);
  servo.write(constrain(a, 0, 180));
  delay(servoEsperaMs);
  servo.detach();
  incAtual = inc;
}

// ---------- Medição ----------
float velSom() { return (331.3 + 0.606 * temperatura) / 10000.0; }

float medirUma() {
  digitalWrite(PINO_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PINO_TRIG, LOW);
  unsigned long d = pulseIn(PINO_ECHO, HIGH, 30000UL);
  return (d == 0) ? -1 : d * velSom() / 2.0 + distOffset;
}

// Distância ao longo do raio até à placa (para cortar o eco dela)
float distPlaca(int inc) {
  float s = sin(radians(inc));
  return (s > 0.01) ? H_SENSOR / s : 999;
}

// Distância ao longo do raio até ao eixo do prato
float distEixo(int inc) {
  return R_SENSOR / cos(radians(inc));
}

float distMaxPara(int inc) {
  if (!janelaAuto) return distMaxManual;
  return min(distPlaca(inc) - MARGEM_PRATO, distMaxManual);
}

float medirMediana(int inc) {
  float v[MAX_LEITURAS];
  int n = 0;
  float dMax = distMaxPara(inc);
  for (int i = 0; i < nLeituras; i++) {
    float d = medirUma();
    if (d >= distMin && d <= dMax) v[n++] = d;
    delay(40);
  }
  if (n * 2 < nLeituras) return -1;
  for (int i = 1; i < n; i++) {
    float x = v[i]; int j = i - 1;
    while (j >= 0 && v[j] > x) { v[j + 1] = v[j]; j--; }
    v[j + 1] = x;
  }
  return (n % 2) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

// ---------- Alinhamento ----------
void alinhar() {
  Serial.println(F("# Alinhar: poe uma vareta fina no centro do prato."));
  Serial.println(F("# Mexe no portico ate 'medido' ~ 'eixo'. Envia qualquer coisa para sair."));
  while (Serial.available()) Serial.read();
  while (!Serial.available()) {
    float d = medirUma();
    Serial.print(F("inc ")); Serial.print(incAtual);
    Serial.print(F(" | medido: "));
    if (d < 0) Serial.print(F("sem eco"));
    else Serial.print(d, 1);
    Serial.print(F(" | eixo: ")); Serial.print(distEixo(incAtual), 1);
    Serial.print(F(" | placa: ")); Serial.println(distPlaca(incAtual), 1);
    delay(250);
  }
  while (Serial.available()) Serial.read();
  Serial.println(F("# Fim do alinhamento."));
}

// ---------- Varrimento ----------
void imprimir(int k, float ang, int inc, float d) {
  Serial.print(k); Serial.print(',');
  Serial.print(ang, 1); Serial.print(',');
  Serial.print(inc); Serial.print(',');
  Serial.println(d, 2);
}

void varrimento() {
  mostrarConfig();
  Serial.println(F("# posicao,angulo_prato,inclinacao,distancia_cm"));
  long passoAtual = 0;
  unsigned long t0 = millis();

  for (int k = 0; k < nPosicoes; k++) {
    long alvo = lround((float)k * PASSOS_VOLTA / nPosicoes);
    rodarPassos(alvo - passoAtual);
    passoAtual = alvo;
    delay(esperaPratoMs);
    float ang = k * 360.0 / nPosicoes;

    if (usarInclinacao) {
      int nInc = (incMax - incMin) / incPasso;
      bool subir = (k % 2 == 0);
      for (int i = 0; i <= nInc; i++) {
        int inc = subir ? incMin + i * incPasso : incMax - i * incPasso;
        moverServo(inc);
        imprimir(k, ang, inc, medirMediana(inc));
      }
    } else {
      if (incAtual != incFixa) moverServo(incFixa);
      imprimir(k, ang, incFixa, medirMediana(incFixa));
    }
  }

  rodarPassos(PASSOS_VOLTA - passoAtual);   // completa a volta no mesmo sentido
  Serial.print(F("# Fim. Duracao: "));
  Serial.print((millis() - t0) / 1000);
  Serial.println(F(" s"));
}

// ---------- Interface ----------
void mostrarConfig() {
  Serial.println(F("# --- Configuracao ---"));
  Serial.print(F("# Geometria: R=")); Serial.print(R_SENSOR, 1);
  Serial.print(F(" cm, H=")); Serial.print(H_SENSOR, 1); Serial.println(F(" cm"));
  Serial.print(F("# Posicoes: ")); Serial.print(nPosicoes);
  Serial.print(F(" (")); Serial.print(360.0 / nPosicoes, 2); Serial.println(F(" graus)"));
  if (usarInclinacao) {
    Serial.print(F("# Inclinacao: ")); Serial.print(incMin); Serial.print(F(" a "));
    Serial.print(incMax); Serial.print(F(", passo ")); Serial.println(incPasso);
  } else {
    Serial.print(F("# Inclinacao fixa: ")); Serial.println(incFixa);
  }
  Serial.print(F("# Servo: offset ")); Serial.print(servoOffset);
  Serial.print(F(", invertido ")); Serial.println(servoInvertido ? F("sim") : F("nao"));
  Serial.print(F("# Leituras: ")); Serial.print(nLeituras);
  Serial.print(F(" | janela: ")); Serial.print(distMin, 1); Serial.print(F(" a "));
  if (janelaAuto) Serial.println(F("auto (corta a placa)"));
  else { Serial.print(distMaxManual, 1); Serial.println(F(" cm")); }
  Serial.print(F("# Temperatura: ")); Serial.println(temperatura, 1);
}

void mostrarAjuda() {
  Serial.println(F("# --- Comandos ---"));
  Serial.println(F("# (Enter) ou iniciar   varrimento completo"));
  Serial.println(F("# alinhar              distancia em tempo real vs eixo"));
  Serial.println(F("# medir                uma medicao no angulo atual"));
  Serial.println(F("# servo ANG            move o sensor"));
  Serial.println(F("# prato GRAUS          roda o prato"));
  Serial.println(F("# geo R H              geometria em cm"));
  Serial.println(F("# inc MIN MAX PASSO    gama de inclinacao"));
  Serial.println(F("# fixa ANG             sensor fixo"));
  Serial.println(F("# pos N                posicoes por volta"));
  Serial.println(F("# leituras N           leituras por ponto"));
  Serial.println(F("# janela MIN MAX       janela manual | janela auto"));
  Serial.println(F("# offset X | inverter  ajuste do servo"));
  Serial.println(F("# temp T | ver | ajuda"));
}

int lerNumeros(String s, float *a, int max) {
  int n = 0;
  s.trim();
  while (s.length() > 0 && n < max) {
    int sp = s.indexOf(' ');
    a[n++] = ((sp < 0) ? s : s.substring(0, sp)).toFloat();
    if (sp < 0) break;
    s = s.substring(sp + 1);
    s.trim();
  }
  return n;
}

void processarComando(String s) {
  s.trim();
  s.toLowerCase();
  if (s.length() == 0 || s == "iniciar") { varrimento(); return; }

  String cmd = s, args = "";
  int sp = s.indexOf(' ');
  if (sp > 0) { cmd = s.substring(0, sp); args = s.substring(sp + 1); args.trim(); }
  float a[3];
  int na = lerNumeros(args, a, 3);
  bool mostrar = true;

  if (cmd == "alinhar") { alinhar(); mostrar = false; }
  else if (cmd == "medir") {
    float d = medirMediana(incAtual);
    Serial.print(F("# Distancia: "));
    if (d < 0) Serial.println(F("sem eco valido"));
    else { Serial.print(d, 2); Serial.println(F(" cm")); }
    mostrar = false;
  }
  else if (cmd == "servo" && na == 1) { moverServo(a[0]); mostrar = false; Serial.println(F("# OK")); }
  else if (cmd == "prato" && na == 1) {
    rodarPassos(lround(a[0] * PASSOS_VOLTA / 360.0)); mostrar = false; Serial.println(F("# OK"));
  }
  else if (cmd == "geo" && na == 2) { R_SENSOR = a[0]; H_SENSOR = a[1]; }
  else if (cmd == "inc" && na == 3 && a[2] > 0 && a[1] >= a[0]) {
    incMin = a[0]; incMax = a[1]; incPasso = a[2]; usarInclinacao = true;
  }
  else if (cmd == "fixa" && na == 1) { incFixa = a[0]; usarInclinacao = false; }
  else if (cmd == "pos" && na == 1 && a[0] >= 1) nPosicoes = a[0];
  else if (cmd == "leituras" && na == 1) nLeituras = constrain((int)a[0], 1, MAX_LEITURAS);
  else if (cmd == "janela" && args == "auto") janelaAuto = true;
  else if (cmd == "janela" && na == 2 && a[1] > a[0]) {
    distMin = a[0]; distMaxManual = a[1]; janelaAuto = false;
  }
  else if (cmd == "offset" && na == 1) { servoOffset = a[0]; moverServo(incAtual); }
  else if (cmd == "inverter") { servoInvertido = !servoInvertido; moverServo(incAtual); }
  else if (cmd == "temp" && na == 1) temperatura = a[0];
  else if (cmd == "ver") { }
  else if (cmd == "ajuda") { mostrarAjuda(); mostrar = false; }
  else { Serial.println(F("# Comando desconhecido. Escreve 'ajuda'.")); mostrar = false; }

  if (mostrar) mostrarConfig();
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  pinMode(PINO_TRIG, OUTPUT);
  pinMode(PINO_ECHO, INPUT);
  digitalWrite(PINO_TRIG, LOW);
  for (int i = 0; i < 4; i++) pinMode(PINOS_STEPPER[i], OUTPUT);
  desligarBobinas();
  moverServo(incFixa);
  mostrarAjuda();
  mostrarConfig();
}

void loop() {
  if (Serial.available()) processarComando(Serial.readStringUntil('\n'));
}