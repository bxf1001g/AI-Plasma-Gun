/*
  Plasma Gun Controller v3.1 - Bird Scarer for Shrimp Pond
  Hardware (all via relay modules):
    GPIO 6  → Relay → 12V Solenoid Valve (Butane Gas)
    GPIO 21 → Relay → 400kV Spark Plug
    GPIO 3  → Relay → 12V Vacuum Pump 15L/min (purge + air fill)

  Combustion Chamber: 250mm × 100mm ID (~1.96L)
  Barrel:             600mm × 50mm ID  (~1.18L)
  Total volume:       ~3.14L

  Pump physics: 15L/min = 0.25L/sec
  To purge 3.14L of CO₂ need ~13 seconds minimum.
  Pump seals chamber when OFF (no passive air flow).

  Sequence (v3.2 - pump runs continuously): PURGE → GAS+PUMP → SPARK+PUMP → EXHAUST
  Pump stays ON during gas fill AND spark — moving air mixes with gas inline,
  reproducing the manual condition that consistently produces a blast.

  Mobile UI: Connect to "PlasmaGun-AP" → http://192.168.4.1
*/

#include <WiFi.h>
#include <WebServer.h>
#include <DNSServer.h>

// ── Pin Definitions ──────────────────────────────────────────────────────────
#define VALVE_PIN    6   // Solenoid valve (butane gas)
#define SPARK_PIN   21   // Spark plug (400kV ignition)
#define EXHAUST_PIN  3   // Vacuum pump (purge + air fill + exhaust)

// ── Relay Logic ──────────────────────────────────────────────────────────────
// Active-LOW relays: LOW = relay ON, HIGH = relay OFF
#define RELAY_ON   LOW
#define RELAY_OFF  HIGH
const char* AP_SSID = "PlasmaGun-AP";
const char* AP_PASS = "plasma1234";

// ── Timing defaults (ms) — ALL tweakable from the mobile UI ──────────────────
int PRE_FLUSH_TIME  = 15000; // LONG purge: 15s × 0.25L/s = 3.75L (>3.14L chamber+barrel)
int GAS_FILL_TIME   = 225;   // gas burst per pulse (matches working manual test)
int GAS_PULSES      = 1;     // single burst with pump running = inline mixing
int GAS_PAUSE       = 200;   // pause between pulses (only used if GAS_PULSES>1)
int AIR_MIX_TIME    = 0;     // unused — pump runs continuously through gas+spark
int SETTLE_DELAY    = 0;     // 0 = spark immediately while pump still pushing mix
int SPARK_DURATION  = 500;   // each spark attempt duration
int SPARK_RETRIES   = 3;     // number of spark attempts
int SPARK_GAP       = 300;   // gap between spark retries
int EXHAUST_TIME    = 15000; // LONG exhaust: fully purge CO₂ for next shot
int COOLDOWN_MS     = 5000;  // min gap between full sequences

// ── State ────────────────────────────────────────────────────────────────────
WebServer server(80);
DNSServer dns;
volatile bool isFiring = false;
unsigned long lastFire = 0;
String currentStep = "READY";

// ── Non-blocking fire state machine ──────────────────────────────────────────
enum FireState {
  FS_IDLE,
  FS_PURGE,
  FS_GAS_ON,
  FS_GAS_OFF,
  FS_SETTLE,
  FS_SPARK_ON,
  FS_SPARK_OFF,
  FS_EXHAUST,
  FS_DONE
};
FireState fireState = FS_IDLE;
unsigned long stateStartedAt = 0;
int gasPulseIdx = 0;
int sparkRetryIdx = 0;

// ─────────────────────────────────────────────────────────────────────────────
// Mobile UI — full control panel with individual test buttons & timing sliders
// ─────────────────────────────────────────────────────────────────────────────
const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0,user-scalable=no"/>
  <title>Plasma Gun v2</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:#0d0d1a;color:#e0e0ff;font-family:'Segoe UI',sans-serif;
         display:flex;flex-direction:column;align-items:center;padding:24px 16px}
    h1{font-size:1.5rem;color:#a78bfa;letter-spacing:2px}
    .sub{font-size:.8rem;color:#6b7280;margin-bottom:24px}
    .card{background:#1a1a2e;border:1px solid #2d2d4e;border-radius:12px;
          padding:16px;width:100%;max-width:380px;margin-bottom:16px}
    .card h3{font-size:.85rem;color:#a78bfa;margin-bottom:12px}

    /* Status */
    .status{text-align:center;padding:20px}
    .step-label{font-size:.7rem;color:#6b7280;margin-bottom:4px}
    .step-val{font-size:1.2rem;font-weight:700;color:#34d399;min-height:1.5em}
    .step-val.active{color:#facc15;animation:pulse 1s infinite}
    .step-val.bang{color:#f87171;animation:blink .3s 3}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
    @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

    /* Sequence diagram */
    .seq{display:flex;gap:4px;justify-content:center;margin-top:12px;flex-wrap:wrap}
    .seq-step{font-size:.6rem;padding:4px 8px;border-radius:6px;background:#2d2d4e;color:#9ca3af}
    .seq-step.on{background:#7c3aed;color:#fff}
    .seq-step.done{background:#065f46;color:#34d399}

    /* Fire button */
    .fire-wrap{display:flex;justify-content:center;margin:8px 0 16px}
    .fire-btn{width:160px;height:160px;border-radius:50%;border:4px solid #7c3aed;
              background:radial-gradient(circle,#4c1d95,#1e1b4b);color:#fff;
              font-size:1rem;font-weight:700;letter-spacing:2px;cursor:pointer;
              box-shadow:0 0 30px #7c3aed88;transition:all .15s;outline:none;
              -webkit-tap-highlight-color:transparent}
    .fire-btn:active{transform:scale(.93)}
    .fire-btn.disabled{border-color:#374151;background:radial-gradient(circle,#1f2937,#111827);
                       box-shadow:none;cursor:not-allowed;color:#6b7280}
    .fire-btn.boom{border-color:#f87171;background:radial-gradient(circle,#7f1d1d,#1a0000);
                   box-shadow:0 0 60px #f87171cc}

    /* Test buttons */
    .test-row{display:flex;gap:8px;margin-bottom:10px}
    .tbtn{flex:1;padding:10px 6px;border-radius:8px;border:none;font-size:.78rem;
          font-weight:600;cursor:pointer;color:#fff;transition:all .1s}
    .tbtn:active{transform:scale(.95)}
    .tbtn.gas{background:#0d9488}
    .tbtn.spark{background:#dc2626}
    .tbtn.fan{background:#2563eb}
    .tbtn.off{background:#374151}

    /* Sliders */
    .row{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
    .row label{font-size:.78rem;color:#9ca3af;flex:1}
    .row input[type=range]{width:110px;accent-color:#7c3aed}
    .row span{font-size:.78rem;color:#e0e0ff;width:52px;text-align:right}
    .apply{width:100%;padding:10px;border-radius:8px;border:none;background:#7c3aed;
           color:#fff;font-weight:600;font-size:.85rem;cursor:pointer;margin-top:4px}
    .apply:active{background:#5b21b6}

    /* Log */
    .log{font-size:.7rem;color:#6b7280;max-height:120px;overflow-y:auto;margin-top:8px;
         font-family:monospace;line-height:1.6}
    .log .entry{border-bottom:1px solid #2d2d4e;padding:2px 0}
    .log .ts{color:#4b5563}
    .log .ev{color:#a78bfa}

    .toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);
           background:#374151;color:#fff;padding:8px 20px;border-radius:20px;
           font-size:.8rem;opacity:0;transition:opacity .3s;pointer-events:none;z-index:99}
    .toast.show{opacity:1}
  </style>
</head>
<body>
  <h1>⚡ PLASMA GUN v3.1</h1>
  <div class="sub">Chamber: 250×100mm · Barrel: 600×50mm · Pump: 15L/min</div>

  <!-- Status Card -->
  <div class="card status">
    <div class="step-label">CURRENT STEP</div>
    <div class="step-val" id="sv">READY</div>
    <div class="seq" id="seq">
      <div class="seq-step" id="s0">PURGE</div>
      <div class="seq-step" id="s1">GAS FILL</div>
      <div class="seq-step" id="s2">SETTLE</div>
      <div class="seq-step" id="s3">SPARK</div>
      <div class="seq-step" id="s4">EXHAUST</div>
    </div>
  </div>

  <!-- Fire Button -->
  <div class="fire-wrap">
    <button class="fire-btn" id="fireBtn" onclick="fire()">🔥<br>FIRE</button>
  </div>

  <!-- Individual GPIO Test -->
  <div class="card">
    <h3>🧪 TEST INDIVIDUAL COMPONENTS</h3>
    <div class="test-row">
      <button class="tbtn gas" onclick="test('valve','on')">⛽ Gas ON</button>
      <button class="tbtn off" onclick="test('valve','off')">Gas OFF</button>
    </div>
    <div class="test-row">
      <button class="tbtn spark" onclick="test('spark','on')">⚡ Spark ON</button>
      <button class="tbtn off" onclick="test('spark','off')">Spark OFF</button>
    </div>
    <div class="test-row">
      <button class="tbtn fan" onclick="test('fan','on')">💨 Pump ON</button>
      <button class="tbtn off" onclick="test('fan','off')">Pump OFF</button>
    </div>
    <div class="test-row">
      <button class="tbtn off" style="background:#991b1b" onclick="test('all','off')">🛑 ALL OFF (Emergency)</button>
    </div>
  </div>

  <!-- Timing Settings -->
  <div class="card">
    <h3>⏱ SEQUENCE TIMING</h3>
    <div class="row">
      <label>Purge 💨</label>
      <input type="range" id="pf" min="0" max="30000" step="1000" value="15000"
             oninput="lbl('pf','pfl','ms')"/>
      <span id="pfl">15000ms</span>
    </div>
    <div class="row">
      <label>Gas Burst</label>
      <input type="range" id="gf" min="50" max="1000" step="25" value="120"
             oninput="lbl('gf','gfl','ms')"/>
      <span id="gfl">120ms</span>
    </div>
    <div class="row">
      <label>Pulses</label>
      <input type="range" id="gp" min="1" max="10" step="1" value="2"
             oninput="lbl('gp','gpl','×')"/>
      <span id="gpl">2×</span>
    </div>
    <div class="row">
      <label>Pulse Gap</label>
      <input type="range" id="gg" min="100" max="2000" step="50" value="300"
             oninput="lbl('gg','ggl','ms')"/>
      <span id="ggl">300ms</span>
    </div>
    <div class="row">
      <label>Settle</label>
      <input type="range" id="sd" min="500" max="8000" step="250" value="3000"
             oninput="lbl('sd','sdl','ms')"/>
      <span id="sdl">3000ms</span>
    </div>
    <div class="row">
      <label>Spark</label>
      <input type="range" id="sk" min="200" max="2000" step="100" value="500"
             oninput="lbl('sk','skl','ms')"/>
      <span id="skl">500ms</span>
    </div>
    <div class="row">
      <label>Spark Retries</label>
      <input type="range" id="sr" min="1" max="5" step="1" value="3"
             oninput="lbl('sr','srl','×')"/>
      <span id="srl">3×</span>
    </div>
    <div class="row">
      <label>Spark Gap</label>
      <input type="range" id="sg" min="100" max="1000" step="50" value="300"
             oninput="lbl('sg','sgl','ms')"/>
      <span id="sgl">300ms</span>
    </div>
    <div class="row">
      <label>Exhaust</label>
      <input type="range" id="ex" min="1000" max="30000" step="1000" value="15000"
             oninput="lbl('ex','exl','ms')"/>
      <span id="exl">15000ms</span>
    </div>
    <div class="row">
      <label>Cooldown</label>
      <input type="range" id="cd" min="2000" max="20000" step="500" value="5000"
             oninput="lbl('cd','cdl','ms')"/>
      <span id="cdl">5000ms</span>
    </div>
    <button class="apply" onclick="applySettings()">💾 Apply Settings</button>
  </div>

  <!-- Event Log -->
  <div class="card">
    <h3>📋 EVENT LOG</h3>
    <div class="log" id="log"></div>
  </div>

  <div class="toast" id="toast"></div>

<script>
  let busy = false;

  function fire() {
    if (busy) return;
    busy = true;
    const btn = document.getElementById('fireBtn');
    const sv  = document.getElementById('sv');
    btn.classList.add('boom');

    fetch('/fire').then(r=>r.text()).then(m=>{toast(m);addLog('FIRE',m)}).catch(()=>toast('Connection error'));

    const pf = +$('pf').value, gf = +$('gf').value, sd = +$('sd').value;
    const sk = +$('sk').value, ex = +$('ex').value, cd = +$('cd').value;
    const gp = +$('gp').value, gg = +$('gg').value;
    const sr = +$('sr').value, sg = +$('sg').value;
    const gasTotalMs = gf*gp + gg*(gp-1);
    const sparkTotalMs = sk*sr + sg*(sr-1);

    let t = 0;
    setStep('PURGE 💨','s0',t);
    t += pf;
    setStep('GAS FILL ('+gp+'×'+gf+'ms)','s1',t);
    t += gasTotalMs;
    setStep('SETTLE','s2',t);
    t += sd;
    setStep('💥 SPARK ('+sr+'× retries)','s3',t);
    t += sparkTotalMs;
    setStep('EXHAUST','s4',t);
    t += ex;

    setTimeout(()=>{
      sv.textContent = 'COOLDOWN';
      sv.className = 'step-val';
      btn.classList.remove('boom');
      btn.classList.add('disabled');
    }, t);

    setTimeout(()=>{
      sv.textContent = 'READY';
      btn.classList.remove('disabled');
      resetSeq();
      busy = false;
    }, t + cd);
  }

  function setStep(name, id, delay) {
    setTimeout(()=>{
      const sv = document.getElementById('sv');
      sv.textContent = name;
      sv.className = 'step-val active';
      if(name.includes('SPARK')){sv.className='step-val bang'}
      // highlight sequence
      for(let i=0;i<5;i++){
        const el = document.getElementById('s'+i);
        if('s'+i===id) el.className='seq-step on';
        else if(i < parseInt(id[1])) el.className='seq-step done';
      }
    }, delay);
  }

  function resetSeq(){for(let i=0;i<5;i++)document.getElementById('s'+i).className='seq-step'}

  function test(comp, state) {
    fetch(`/test?c=${comp}&s=${state}`).then(r=>r.text()).then(m=>{toast(m);addLog('TEST',m)}).catch(()=>toast('Error'));
  }

  function applySettings() {
    const p = ['pf','gf','gp','gg','sd','sk','sr','sg','ex','cd'].map(id=>id+'='+$(id).value).join('&');
    fetch('/settings?'+p).then(r=>r.text()).then(m=>{toast(m);addLog('SETTINGS',m)}).catch(()=>toast('Error'));
  }

  function lbl(id,lid,u){$(lid).textContent=$(id).value+u}
  function $(id){return document.getElementById(id)}

  function addLog(type, msg) {
    const log = $('log');
    const now = new Date().toLocaleTimeString();
    log.innerHTML = `<div class="entry"><span class="ts">[${now}]</span> <span class="ev">${type}</span> ${msg}</div>` + log.innerHTML;
  }

  function toast(msg) {
    const t = $('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(()=>t.classList.remove('show'), 2500);
  }
</script>
</body>
</html>
)rawliteral";

// ─────────────────────────────────────────────────────────────────────────────
// Fire Sequence — NON-BLOCKING state machine
// loop() must call fireTick() every iteration. Pump (EXHAUST_PIN) stays ON
// continuously from PURGE through EXHAUST, only switching off at the very end.
// ─────────────────────────────────────────────────────────────────────────────
void startFireSequence() {
  isFiring = true;
  lastFire = millis();
  gasPulseIdx = 0;
  sparkRetryIdx = 0;

  // Pump ON for the entire cycle
  digitalWrite(EXHAUST_PIN, RELAY_ON);

  currentStep = "PURGE";
  fireState = FS_PURGE;
  stateStartedAt = millis();
}

void fireTick() {
  if (fireState == FS_IDLE || fireState == FS_DONE) return;
  unsigned long elapsed = millis() - stateStartedAt;

  switch (fireState) {

    case FS_PURGE:
      if (elapsed >= (unsigned long)PRE_FLUSH_TIME) {
        currentStep = "GAS FILL";
        digitalWrite(VALVE_PIN, RELAY_ON);
        fireState = FS_GAS_ON;
        stateStartedAt = millis();
      }
      break;

    case FS_GAS_ON:
      if (elapsed >= (unsigned long)GAS_FILL_TIME) {
        digitalWrite(VALVE_PIN, RELAY_OFF);
        gasPulseIdx++;
        if (gasPulseIdx < GAS_PULSES) {
          fireState = FS_GAS_OFF;
          stateStartedAt = millis();
        } else {
          // done with gas pulses → SETTLE (or SPARK if SETTLE_DELAY=0)
          if (SETTLE_DELAY > 0) {
            currentStep = "SETTLE";
            fireState = FS_SETTLE;
          } else {
            currentStep = "SPARK";
            digitalWrite(SPARK_PIN, RELAY_ON);
            fireState = FS_SPARK_ON;
          }
          stateStartedAt = millis();
        }
      }
      break;

    case FS_GAS_OFF:
      if (elapsed >= (unsigned long)GAS_PAUSE) {
        digitalWrite(VALVE_PIN, RELAY_ON);
        fireState = FS_GAS_ON;
        stateStartedAt = millis();
      }
      break;

    case FS_SETTLE:
      if (elapsed >= (unsigned long)SETTLE_DELAY) {
        currentStep = "SPARK";
        digitalWrite(SPARK_PIN, RELAY_ON);
        fireState = FS_SPARK_ON;
        stateStartedAt = millis();
      }
      break;

    case FS_SPARK_ON:
      if (elapsed >= (unsigned long)SPARK_DURATION) {
        digitalWrite(SPARK_PIN, RELAY_OFF);
        sparkRetryIdx++;
        if (sparkRetryIdx < SPARK_RETRIES) {
          fireState = FS_SPARK_OFF;
          stateStartedAt = millis();
        } else {
          currentStep = "EXHAUST";
          fireState = FS_EXHAUST;
          stateStartedAt = millis();
        }
      }
      break;

    case FS_SPARK_OFF:
      if (elapsed >= (unsigned long)SPARK_GAP) {
        digitalWrite(SPARK_PIN, RELAY_ON);
        fireState = FS_SPARK_ON;
        stateStartedAt = millis();
      }
      break;

    case FS_EXHAUST:
      if (elapsed >= (unsigned long)EXHAUST_TIME) {
        digitalWrite(EXHAUST_PIN, RELAY_OFF);  // pump OFF only here
        currentStep = "READY";
        isFiring = false;
        fireState = FS_DONE;
      }
      break;

    default:
      break;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP Handlers
// ─────────────────────────────────────────────────────────────────────────────
void handleRoot() {
  server.send_P(200, "text/html", INDEX_HTML);
}

void handleFire() {
  if (isFiring) { server.send(200, "text/plain", "Busy: " + currentStep); return; }
  unsigned long now = millis();
  if (lastFire != 0 && now - lastFire < (unsigned long)COOLDOWN_MS) {
    server.send(200, "text/plain", "Cooling down...");
    return;
  }
  server.send(200, "text/plain", "Sequence started!");
  startFireSequence();
}

void handleTest() {
  String comp  = server.arg("c");
  String state = server.arg("s");
  int val = (state == "on") ? RELAY_ON : RELAY_OFF;

  if (comp == "valve")    { digitalWrite(VALVE_PIN, val); }
  else if (comp == "spark") { digitalWrite(SPARK_PIN, val); }
  else if (comp == "fan")   { digitalWrite(EXHAUST_PIN, val); }
  else if (comp == "all")   {
    digitalWrite(VALVE_PIN, RELAY_OFF);
    digitalWrite(SPARK_PIN, RELAY_OFF);
    digitalWrite(EXHAUST_PIN, RELAY_OFF);
  }
  server.send(200, "text/plain", comp + " → " + state);
}

void handleSettings() {
  if (server.hasArg("pf")) PRE_FLUSH_TIME = server.arg("pf").toInt();
  if (server.hasArg("gf")) GAS_FILL_TIME  = server.arg("gf").toInt();
  if (server.hasArg("gp")) GAS_PULSES     = server.arg("gp").toInt();
  if (server.hasArg("gg")) GAS_PAUSE      = server.arg("gg").toInt();
  if (server.hasArg("sd")) SETTLE_DELAY   = server.arg("sd").toInt();
  if (server.hasArg("sk")) SPARK_DURATION = server.arg("sk").toInt();
  if (server.hasArg("sr")) SPARK_RETRIES  = server.arg("sr").toInt();
  if (server.hasArg("sg")) SPARK_GAP      = server.arg("sg").toInt();
  if (server.hasArg("ex")) EXHAUST_TIME   = server.arg("ex").toInt();
  if (server.hasArg("cd")) COOLDOWN_MS    = server.arg("cd").toInt();

  String msg = "Saved! Gas:" + String(GAS_FILL_TIME) + "ms×" + String(GAS_PULSES)
             + " Spark:" + String(SPARK_DURATION) + "ms×" + String(SPARK_RETRIES)
             + " Purge:" + String(PRE_FLUSH_TIME) + "ms";
  server.send(200, "text/plain", msg);
}

void handleStatus() {
  String json = "{\"firing\":" + String(isFiring ? "true" : "false")
              + ",\"step\":\"" + currentStep + "\"}";
  server.send(200, "application/json", json);
}

// ─────────────────────────────────────────────────────────────────────────────
// Setup & Loop
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(VALVE_PIN, OUTPUT);
  pinMode(SPARK_PIN, OUTPUT);
  pinMode(EXHAUST_PIN, OUTPUT);
  // Active-LOW relays: start with all relays OFF
  digitalWrite(VALVE_PIN, RELAY_OFF);
  digitalWrite(SPARK_PIN, RELAY_OFF);
  digitalWrite(EXHAUST_PIN, RELAY_OFF);

  WiFi.softAP(AP_SSID, AP_PASS);
  IPAddress ip = WiFi.softAPIP();

  Serial.println("\n=== Plasma Gun v3.1 (Pump) ===");
  Serial.println("GPIO 6  → Solenoid Valve");
  Serial.println("GPIO 21 → Spark Plug");
  Serial.println("GPIO 3  → Vacuum Pump 15L/min");
  Serial.print("WiFi: "); Serial.println(AP_SSID);
  Serial.print("URL : http://"); Serial.println(ip);
  Serial.println("==============================\n");

  server.on("/",         handleRoot);
  server.on("/fire",     handleFire);
  server.on("/test",     handleTest);
  server.on("/settings", handleSettings);
  server.on("/status",   handleStatus);

  // Captive portal redirects
  server.onNotFound([]() {
    server.sendHeader("Location", "http://192.168.4.1", true);
    server.send(302, "text/plain", "");
  });
  server.on("/hotspot-detect.html",        [](){ server.sendHeader("Location","http://192.168.4.1",true); server.send(302,"text/plain",""); });
  server.on("/library/test/success.html",  [](){ server.sendHeader("Location","http://192.168.4.1",true); server.send(302,"text/plain",""); });
  server.on("/generate_204",               [](){ server.sendHeader("Location","http://192.168.4.1",true); server.send(302,"text/plain",""); });
  server.on("/gen_204",                    [](){ server.sendHeader("Location","http://192.168.4.1",true); server.send(302,"text/plain",""); });
  server.on("/connecttest.txt",            [](){ server.sendHeader("Location","http://192.168.4.1",true); server.send(302,"text/plain",""); });

  server.begin();
  dns.start(53, "*", WiFi.softAPIP());

  Serial.println("Ready to fire!");
}

void loop() {
  dns.processNextRequest();
  server.handleClient();
  fireTick();
}
