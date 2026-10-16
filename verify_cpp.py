#!/usr/bin/env python3
"""
Validate the embedded C++ tasks by compiling and running a correct reference
implementation against each task's assertions.

A benchmark whose tests are wrong is worse than no benchmark: it reports every
model as failing, which looks like a finding and is actually a harness bug. So
each task is proved here before any model is run against it.

Also checks the failure direction: deliberately corrupt each reference and confirm
the test catches it, so we know the assertion has teeth.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_electronics import MOCK_HEADER  # noqa: E402

TMP = tempfile.mkdtemp(prefix="embc-verify-")

# Reference implementations. Deliberately plain - the point is to satisfy the
# spec, not to be idiomatic.
REF = {
"embc/millis-rollover": """
bool elapsed(uint32_t now, uint32_t last, uint32_t interval) {
    return (uint32_t)(now - last) >= interval;
}
""",
"embc/debounce": """
#include <cstdint>
#define DEBOUNCE_MS 20
struct ButtonDebouncer {
    uint32_t last_change;
    bool stable;
    bool candidate;
    bool init = false;
    bool update(uint32_t now, bool raw) {
        if (!init) { init = true; stable = raw; candidate = raw; last_change = now; return stable; }
        if (raw != candidate) { candidate = raw; last_change = now; }
        if ((uint32_t)(now - last_change) >= DEBOUNCE_MS) stable = candidate;
        return stable;
    }
};
""",
"embc/pwm-timing": """
#include <cstdint>
uint32_t highTimeMicros(uint32_t dutyPermille, uint32_t periodMicros) {
    if (dutyPermille == 0) return 0;
    if (dutyPermille > 1000) dutyPermille = 1000;
    return (uint32_t)((uint64_t)periodMicros * dutyPermille / 1000ULL);
}
""",
"embc/moving-average": """
#include <cstdint>
#include <cstddef>
struct MovingAvg {
    int16_t buf[8]; size_t head = 0, n = 0;
    void push(int16_t v) { buf[head] = v; head = (head + 1) % 8; if (n < 8) n++; }
    size_t count() const { return n; }
    int16_t value() const {
        int32_t s = 0; for (size_t i = 0; i < n; i++) s += buf[i];
        return (int16_t)(s / (int32_t)n);
    }
};
""",
"embc/median-filter": """
#include <cstdint>
int16_t median3(int16_t a, int16_t b, int16_t c) {
    if (a > b) { int16_t t = a; a = b; b = t; }
    if (b > c) { int16_t t = b; b = c; c = t; }
    if (a > b) { int16_t t = a; a = b; b = t; }
    return b;
}
""",
"embc/pid": """
#include <cstdint>
struct PID {
    float kp, ki, kd, integral = 0.0f, last_error = 0.0f;
    uint32_t last_time = 0; bool first = true;
    float update(float setpoint, float measured, uint32_t now_ms) {
        float err = setpoint - measured;
        if (first) { last_time = now_ms; first = false; }
        float dt = (float)(now_ms - last_time) / 1000.0f;
        last_time = now_ms;
        integral += err * (dt > 0.0f ? dt : 0.0f);
        if (integral > 1000.0f) integral = 1000.0f;
        if (integral < -1000.0f) integral = -1000.0f;
        float deriv = (dt > 0.0f) ? (err - last_error) / dt : 0.0f;
        last_error = err;
        return kp * err + ki * integral + kd * deriv;
    }
};
""",
"embc/adc-plausibility": """
bool sensorReadingSane(int rawAdc, float volts, float expected, float tolerance) {
    if (rawAdc < 0 || rawAdc > 1023) return false;
    if (volts < 0.0f) return false;
    float d = volts - expected; if (d < 0) d = -d;
    return d <= tolerance;
}
""",
"embc/i2c-register-write": """
#include <cstdint>
#include <cstddef>
size_t encodeRegisterWrite(uint8_t devAddr, uint8_t reg, uint16_t value, uint8_t* out) {
    if (!out) return 0;
    out[0] = devAddr;
    out[1] = (uint8_t)(reg >> 8);
    out[2] = (uint8_t)(reg & 0xFF);
    out[3] = (uint8_t)(value >> 8);
    out[4] = (uint8_t)(value & 0xFF);
    return 5;
}
""",
"embc/state-machine": """
#include <cstdint>
enum class BtnEvent { None, ShortPress, LongPress, DoublePress };
struct ButtonSM {
    bool prev = false;
    uint32_t down_at = 0;
    uint32_t last_release = 0;
    bool long_fired = false;
    bool armed = false;
    BtnEvent update(uint32_t now, bool pressed, uint32_t long_ms, uint32_t dbl_ms) {
        BtnEvent ev = BtnEvent::None;
        if (pressed && !prev) {
            if (armed && (uint32_t)(now - last_release) <= dbl_ms) {
                ev = BtnEvent::DoublePress; armed = false;
            } else {
                armed = false;
            }
            down_at = now; long_fired = false;
        } else if (!pressed && prev) {
            if (!long_fired && (uint32_t)(now - down_at) < long_ms) {
                ev = BtnEvent::ShortPress; last_release = now; armed = true;
            }
        } else if (pressed && prev && !long_fired &&
                   (uint32_t)(now - down_at) >= long_ms) {
            ev = BtnEvent::LongPress; long_fired = true;
        }
        prev = pressed;
        return ev;
    }
};
""",
"embc/watchdog-refresh": """
#include <cstdint>
bool refreshWatchdog(uint32_t now, uint32_t last, uint32_t timeout_ms) {
    uint32_t d = (uint32_t)(now - last);
    if (d >= timeout_ms) return true;
    return false;
}
""",
"embc/serial-parse": """
#include <cstddef>
#include <cstdlib>
#include <cstdint>
bool parseCsvLine(const char* line, float* out, size_t maxOut) {
    if (!line || !out || maxOut == 0) return false;
    if (*line == 0) return false;
    const char* p = line; size_t n = 0;
    while (*p) {
        char* end = nullptr;
        float v = strtof(p, &end);
        if (end == p) return false;
        if (n >= maxOut) return false;
        out[n++] = v;
        if (*end == 0) break;
        if (*end != ',') return false;
        p = end + 1;
    }
    return n > 0;
}
""",
"embc/crc16-modbus": """
#include <cstdint>
#include <cstddef>
uint16_t crc16Modbus(const uint8_t* data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i];
        for (int b = 0; b < 8; b++) {
            if (crc & 1) crc = (uint16_t)((crc >> 1) ^ 0xA001);
            else crc = (uint16_t)(crc >> 1);
        }
    }
    return crc;
}
""",
}

# Corruptions that a wrong-but-plausible implementation would produce. Each must
# make the test FAIL; if a corruption still passes, the assertion is too weak.
CORRUPT = {
    "embc/millis-rollover": "bool elapsed(uint32_t now, uint32_t last, uint32_t interval){\n"
                            "  return (int32_t)(now - last) >= (int32_t)interval; }\n",
    "embc/pwm-timing": "uint32_t highTimeMicros(uint32_t d, uint32_t p){\n"
                       "  return (uint32_t)p * d / 1000; }\n",
    "embc/median-filter": "#include <cstdint>\n"
                          "int16_t median3(int16_t a,int16_t b,int16_t c){ if(a>b){int16_t t=a;a=b;b=t;}"
                          " if(b>c){int16_t t=b;b=c;c=t;} return b; }\n",
    # drops the separator validation: "1;2" would be accepted as two fields
    "embc/serial-parse": "#include <cstddef>\n#include <cstdlib>\n"
                         "bool parseCsvLine(const char* line, float* out, size_t m){\n"
                         "  if(!line||*line==0) return false; size_t n=0; const char* p=line;\n"
                         "  while(*p){ char* e=nullptr; float v=strtof(p,&e); if(e==p) return false;\n"
                         "    if(n>=m) return false; out[n++]=v; if(*e==0) break; p=e+1; }\n"
                         "  return n>0; }\n",
    "embc/crc16-modbus": "#include <cstdint>\n#include <cstddef>\n"
                         "uint16_t crc16Modbus(const uint8_t* d, size_t n){ uint16_t c=0xFFFF;\n"
                         "  for(size_t i=0;i<n;i++){ c^=d[i]; for(int b=0;b<8;b++){ if(c&1) c=(c>>1)^0x8005; else c>>=1; } }\n"
                         "  return c; }\n",
    "embc/i2c-register-write": "#include <cstdint>\n#include <cstddef>\n"
                               "size_t encodeRegisterWrite(uint8_t d, uint8_t r, uint16_t v, uint8_t* o){\n"
                               "  o[0]=d; o[1]=r; o[2]=(uint8_t)(v>>8); o[3]=(uint8_t)(v&0xFF); return 4; }\n",
    "embc/adc-plausibility": "bool sensorReadingSane(int a, float v, float e, float t){\n"
                             "  if(a<0||a>1023) return false; return v>=0.0f; }\n",
    "embc/debounce": "#include <cstdint>\n#define DEBOUNCE_MS 20\n"
                    "struct ButtonDebouncer { bool stable=false;\n"
                    "  bool update(uint32_t now, bool raw){ stable=raw; return stable; } };\n",
    "embc/watchdog-refresh": "#include <cstdint>\n"
                             "bool refreshWatchdog(uint32_t now, uint32_t last, uint32_t t){\n"
                             "  return (int32_t)(now - last) >= (int32_t)t; }\n",
    # returns the mean of the first 8 slots forever, ignoring the head
    "embc/moving-average": "#include <cstdint>\n#include <cstddef>\n"
                           "struct MovingAvg { int16_t buf[8]={0}; size_t head=0, n=0;\n"
                           "  void push(int16_t v){ buf[head]=v; head=(head+1)%8; if(n<8) n++; }\n"
                           "  size_t count() const { return n; }\n"
                           "  int16_t value() const { int32_t s=0; for(size_t i=0;i<8;i++) s+=buf[i]; return (int16_t)(s/8); } };\n",
    # ignores the tolerance check entirely
    "embc/pid": "#include <cstdint>\n"
                "struct PID { float kp,ki,kd,integral=0.0f,last_error=0.0f; uint32_t last_time=0; bool first=true;\n"
                "  float update(float sp, float m, uint32_t now){ float e=sp-m;\n"
                "    integral += e; if(integral>1000) integral=1000; if(integral<-1000) integral=-1000;\n"
                "    return kp*e + ki*integral + kd*(e-last_error); } };\n",
}


def build(tid, body, test):
    src = MOCK_HEADER + "\n" + body + "\n\nint main() {\n" + test + "\n    return 0;\n}\n"
    p = os.path.join(TMP, tid.replace("/", "_") + ".cpp")
    open(p, "w").write(src)
    return p


def compile_run(path):
    exe = path[:-4]
    c = subprocess.run(["g++", "-std=c++17", "-O1", "-w", "-o", exe, path],
                       capture_output=True, text=True, timeout=90)
    if c.returncode != 0:
        return "compile_error", (c.stderr or "")[-500:]
    try:
        r = subprocess.run([exe], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return "timeout", ""
    return ("ok" if r.returncode == 0 else "assert_failed"), (r.stdout or "")[-300:] + (r.stderr or "")[-300:]


def main():
    tasks = json.load(open(os.path.join(HERE, "tasks", "embedded_cpp.json")))
    ok = bad = 0
    for t in tasks:
        tid = t["id"]
        if tid not in REF:
            print(f"  {tid:<32} NO REFERENCE - cannot verify")
            bad += 1
            continue
        st, msg = compile_run(build(tid, REF[tid], t["test"]))
        if st == "ok":
            ok += 1
            note = ""
            if tid in CORRUPT:
                cst, _ = compile_run(build(tid, CORRUPT[tid], t["test"]))
                note = f"  [corruption -> {cst}]" + ("  <-- TEST TOO WEAK" if cst == "ok" else "")
            print(f"  {tid:<32} reference {st}{note}")
        else:
            bad += 1
            print(f"  {tid:<32} reference {st}  <-- BAD TEST")
            print(f"      {msg.strip()[:220]}")
    print(f"\n{ok} verified, {bad} broken")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())