#!/usr/bin/env python3
"""
Electronics / embedded systems category.

Adds to the chemistry and medicine coverage the suite already had. Two parts:

1. `electronics.json` - professional knowledge as multiple choice. Scored from a
   single forward pass via logprobs, so ~50 questions cost ~50 generated tokens
   instead of ~50 generations. At 3-8 tok/s that is the difference between a
   category that runs in two minutes and one that runs for two hours.

   Questions are calibrated to professional practice: register-level facts a
   working embedded engineer uses, not trivia and not graduate-level device
   physics. GPQA is deliberately excluded for the same reason.

2. `embedded_cpp.json` - C++ that actually compiles and runs. A mock Arduino /
   ESP32 / Linux-GPIO header supplies digitalWrite, pinMode, millis, INPUT_PULLUP
   and the rest, so a model's output is verified by execution rather than by a
   reviewer liking it.

   There is no AVR cross-compiler on this box, so a model cannot be scored on
   whether its sketch fits in 2 KB of SRAM or whether `analogWrite` exists on its
   core version. What CAN be scored, and is what actually breaks in practice, is
   the logic: debounce, rollover-safe millis arithmetic, ring buffers, state
   machines, PWM timing, PID, sensor plausibility. Those are compiled with g++ 16
   and executed against assertions.

   The millis-rollover task is included deliberately. `(now - last) >= interval`
   is the single most common embedded bug, it is invisible to a human reviewer
   reading the code, and only a test that simulates the 32-bit wrap finds it.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import debias_mc  # noqa: E402

OUT = os.path.join(HERE, "tasks")

# (id, question, [choices], gold_index, subject)
ELECTRONICS = [
    # ------------------------------------------------------ Arduino / AVR core
    ("ard/avr-sram", "How much SRAM does the ATmega328P on the Arduino Uno have?",
     ["512 bytes", "2 KB", "8 KB", "32 KB"], 1, "arduino"),
    ("ard/f-macro", "On AVR, what does the Arduino F() macro do?",
     ["Forces a float conversion", "Stores a string literal in flash and reads it as a pointer",
      "Disables interrupts briefly", "Formats a string with printf"], 1, "arduino"),
    ("ard/adc-resolution", "The ATmega328P ADC is 10-bit. How many distinct readings can it return?",
     ["256", "512", "1024", "4096"], 2, "arduino"),
    ("ard/adc-reference", "What is the ADC reference voltage on an unmodified Arduino Uno?",
     ["3.3 V", "5 V (AVCC)", "1.1 V internal bandgap", "It is configurable in software"], 1, "arduino"),
    ("ard/volatile-isr", "Why must a variable shared between an ISR and loop() be declared volatile?",
     ["So it survives a power cycle", "So the compiler re-reads it instead of caching it in a register",
      "So it is placed in SRAM rather than flash", "To make access atomic"], 1, "arduino"),
    ("ard/input-pullup", "What does pinMode(pin, INPUT_PULLUP) do?",
     ["Enables an internal 20-50 kOhm pull-up to VCC and leaves the pin high when nothing connects",
      "Enables a pull-down to GND", "Configures an internal ADC channel",
      "Sets the pin to push-pull output"], 0, "arduino"),
    # The pin must be named: analogWrite() uses Timer0 on most pins but Timer2 on
    # pins 3 and 11, and tone() also uses Timer2, so "which timer does
    # analogWrite() use" had two correct answers depending on the pin.
    ("ard/pwm-timer0", "On the Uno, analogWrite() on pin 9 uses which timer, and what else depends on it?",
     ["Timer1, shared with the servo library", "Timer0, which also drives millis() and micros()",
      "Timer2, which also drives tone()", "A software timer, so nothing else is affected"], 1, "arduino"),
    ("ard/delay-bad", "Why is delay() considered bad practice in a loop()?",
     ["It is too imprecise", "It blocks the CPU so nothing else runs, including pin updates and serial receive",
      "It uses too much RAM", "It is not supported on the Uno"], 1, "arduino"),
    ("ard/eeprom-endurance", "The ATmega328P EEPROM is rated for roughly how many write cycles per cell?",
     ["1,000", "100,000", "1,000,000", "10,000,000"], 1, "arduino"),
    # Gold is SPCR. On the ATmega328P the prescaler field SPR1:SPR0 lives in
    # SPCR; SPSR holds only the SPI2X double-speed bit. This was golded on SPSR,
    # and a live model answered SPCR - the datasheet-correct register - at 0.968
    # confidence and was scored wrong. The question is also reworded to name the
    # bits, so it cannot be read as "which register participates in clock setup".
    ("ard/spi-divider", "On AVR, which register holds the SPI clock prescaler bits SPR1 and SPR0?",
     ["SPCR", "SPSR", "PORTB", "ADCSRA"], 0, "arduino"),
    ("ard/i2c-address-7bit", "Datasheets quote I2C addresses as 7-bit. Why do some libraries expect 8 bits?",
     ["They include the read/write bit shifted in", "They are for 8-bit microcontrollers",
      "They mask the top bit", "They use a different address base"], 0, "arduino"),
    ("ard-bootloader-flash", "On the ATmega328P, how much flash is reserved for the bootloader?",
     ["0.5 KB", "1 KB", "2 KB", "4 KB"], 2, "arduino"),

    # ------------------------------------------------------------- Raspberry Pi
    ("rpi/logic-level", "What is the GPIO logic level voltage on a Raspberry Pi?",
     ["5 V", "3.3 V", "1.8 V", "It varies by pin"], 1, "raspberry_pi"),
    ("rpi/5v-danger", "What happens if you drive a Raspberry Pi GPIO pin with a 5 V signal?",
     ["Nothing, the pins are 5 V tolerant", "It works at reduced noise immunity",
      "The 3.3 V SoC is at risk of permanent damage", "It automatically level-shifts"], 2, "raspberry_pi"),
    ("rpi/hw-pwm-pins", "Which GPIO numbers provide hardware PWM on most Raspberry Pi models?",
     ["2, 3, 4, 17", "12, 13, 18, 19", "20, 21, 22, 23", "24, 25, 26, 27"], 1, "raspberry_pi"),
    ("rpi/i2c-pullups", "Which physical pins carry the I2C bus's factory-fitted pull-up resistors?",
     ["Pins 2 and 3 (GPIO0 and GPIO1)", "Pins 24 and 26", "Pins 6 and 9", "Pins 11 and 13"], 0, "raspberry_pi"),
    ("rpi/gpiomem", "On a Pi 4 or later, what must a non-root user join to use the GPIO sysfs interface?",
     ["The sudo group", "The gpio group", "The i2c group", "The spi group"], 1, "raspberry_pi"),
    ("rpi/undervoltage", "A Pi 4 showing a 'undervoltage' warning is most likely because",
     ["The ambient temperature is too high", "The 5 V supply cannot supply 3 A and the SoC is being throttled",
      "The SD card is failing", "The GPIO pins are overloaded"], 1, "raspberry_pi"),
    ("rpi/hats-i2c-conflict", "Why can an I2C HAT that has an EEPROM fail to be detected on some Pi models?",
     ["The EEPROM is on the same I2C bus the camera or CSI uses, at a fixed address",
      "The Pi has only one I2C peripheral", "The EEPROM needs 5 V",
      "The hat must be on the SPI bus"], 0, "raspberry_pi"),
    ("rpi/pwm-software", "Why do most Raspberry Pi GPIOs not produce accurate hardware PWM?",
     ["They have no PWM capability at all", "PWM must be emulated in software (timed loops), which is jittery",
      "They require an external clock", "PWM is only available on the 40-pin header"], 1, "raspberry_pi"),
    # Asked "what do BCM0 and GPIO17 refer to?" and golded "the physical pin number
    # and the Broadcom SoC number". That is wrong: gpio readall's columns are
    # BCM | wPi | Name | Mode | V | Physical, so GPIO17 (the Name column) and
    # BCM0 are BOTH Broadcom numbers - they are not a physical-vs-SoC pair, and
    # the physical pin for GPIO17 is 11. Replaced with the unambiguous mapping.
    ("rpi/logical-vs-physical", "On the Raspberry Pi 40-pin header, GPIO17 is on which physical pin?",
     ["pin 3", "pin 11", "pin 17", "pin 40"], 1, "raspberry_pi"),

    # ------------------------------------------------------------------ ESP32
    ("esp32/adc2-wifi", "On the ESP32, why is ADC2 unusable while Wi-Fi is active?",
     ["ADC2 is reserved for the flash chip", "ADC2 shares hardware with the Wi-Fi RF subsystem",
      "ADC2 draws too much current", "ADC2 must be enabled by a driver call"], 1, "esp32"),
    ("esp32/logic-level", "Is the ESP32's GPIO 5 V tolerant?",
     ["Yes, on all pins", "Yes, on pins 12-19", "No, the ESP32 is 3.3 V only", "Only when the flash is in QIO mode"], 2, "esp32"),
    ("esp32/adc-nonlinear", "The ESP32's ADC response is non-linear at the extremes. What is the consequence?",
     ["It must always be calibrated against a known reference", "It cannot be read at all below 0.3 V",
      "It has only 8 bits of resolution", "It requires an external reference chip"], 0, "esp32"),
    ("esp32/strapping-pins", "What is special about ESP32 strapping pins (GPIO0, 2, 5, 12, 15)?",
     ["They cannot be used for input", "Their state at reset affects boot mode and must be set correctly",
      "They are 5 V tolerant", "They are PWM-only"], 1, "esp32"),
    ("esp32/deepsleep", "Typical ESP32 deep-sleep current is on the order of",
     ["1 microampere", "10 microamperes", "1 milliampere", "50 milliamperes"], 1, "esp32"),
    ("esp32/touch", "Which peripheral gives the ESP32 its capacitive touch inputs?",
     ["The touch sensor peripheral", "The ADC", "The I2C master", "The PCNT unit"], 0, "esp32"),
    ("esp32/watchdog", "What does the ESP32 task watchdog (TWDT) protect against?",
     ["Brownout events", "Tasks that deadlock or block, so the CPU is restarted",
      "Flash wear", "Wi-Fi disconnection"], 1, "esp32"),
    ("esp32/psram", "What problem does external PSRAM solve on an ESP32 module?",
     ["It replaces the flash storage", "It expands the DRAM available to Wi-Fi, buffers and large buffers",
      "It speeds up the CPU clock", "It adds more ADC channels"], 1, "esp32"),

    # ------------------------------------------------------- buses and I/O
    ("bus/open-drain", "An open-drain output can do which two things?",
     ["Drive HIGH or LOW", "Pull LOW or let go (float)", "Drive HIGH only", "Tri-state with strong drive"], 1, "io"),
    ("bus/i2c-pullup-value", "For a 3.3 V I2C bus with 1 mA sink current, the pull-up resistor should be about",
     ["100 ohm", "3.3 kOhm", "10 kOhm", "1 MOhm"], 1, "io"),
    ("bus/rs485", "What is the main advantage of RS-485 over RS-232?",
     ["It is cheaper", "Differential signalling gives noise immunity and allows long runs",
      "It needs only one wire", "It is point-to-point"], 1, "io"),
    ("bus/can-terminators", "CAN bus termination resistors should be placed",
     ["At every node", "120 Ohm at each of the two physical ends only", "In series with the transceiver",
      "Only on the dominant wire"], 1, "io"),
    ("bus/current-loop", "The main advantage of a 4-20 mA current loop over a voltage signal is",
     ["It is cheaper", "It is immune to noise and can power the sensor from the loop",
      "It has lower latency", "It is digital"], 1, "io"),
    ("bus/modbus-registry", "In Modbus, function code 3 reads what?",
     ["Coils", "Discrete inputs", "Holding registers", "Input registers"], 2, "io"),
    ("bus/one-wire", "1-Wire devices share a single signal line plus ground, and the bus needs",
     ["A pull-up resistor to define the idle HIGH", "A separate clock wire", "A differential pair", "An external crystal"], 0, "io"),
    ("bus/spi-vs-i2c", "Compared to I2C, SPI typically achieves higher throughput because",
     ["It uses fewer pins", "It has a separate clock line and separate chip-select lines, so it has no arbitration or stretching",
      "It runs at lower voltage", "It is fully duplex with no pull-ups needed"], 1, "io"),
    ("bus/level-shifter-bss138", "A BSS138 bidirectional level shifter works for I2C because",
     ["It is a high-speed comparator", "It uses MOSFETs to pull the open-drain line to whichever side is higher",
      "It replicates the clock at 5 V", "It boosts the pull-up current"], 1, "io"),
    ("bus/watchdog-purpose", "A hardware watchdog timer does what?",
     ["Logs timestamps", "Resets the MCU if not periodically serviced, recovering from a hang",
      "Increases clock speed under load", "Powers down peripherals between cycles"], 1, "io"),
    ("bus/shmitt", "Why put a Schmitt trigger between a noisy digital source and a microcontroller input?",
     ["To increase drive current", "To provide hysteresis so noise near the threshold cannot cause chatter",
      "To convert the voltage level", "To add a pull-up resistor"], 1, "io"),

    # -------------------------------------------------------------- power
    ("pwr/decoupling", "The 100 nF ceramic capacitor placed at each IC power pin is there to",
     ["Set the DC level", "Supply transient current locally and suppress high-frequency noise",
      "Prevent ESD damage to the pin", "Increase the supply voltage"], 1, "power"),
    ("pwr/ldo-vs-buck", "Compared with an LDO, a buck converter's advantage is",
     ["Lower output noise", "Higher efficiency at large voltage drops, at the cost of switching noise",
      "Simpler layout", "No inductor required"], 1, "power"),
    ("pwr/ldo-dissipation", "An LDO converting 5 V to 3.3 V at 300 mA dissipates approximately",
     ["0.3 W", "0.5 W", "1.5 W", "1.65 W"], 1, "power"),
    ("pwr/flyback", "A diode placed across a relay or solenoid coil is required because",
     ["The coil needs a return path for the reverse EMF when the field collapses",
      "It reduces the coil's resistance", "It protects the driver from ESD", "It speeds up the release"], 0, "power"),
    ("pwr/usb2-limit", "A USB 2.0 port provides a maximum of",
     ["100 mA", "500 mA", "900 mA", "2.4 A"], 1, "power"),
    ("pwr/inrush", "A large capacitor bank switched onto a supply causes inrush current. What limits it?",
     ["A series fuse only", "NTC thermistors, soft-start, or a series resistor",
      "A parallel capacitor", "Raising the supply voltage"], 1, "power"),
    ("pwr/ground-loop", "A ground loop causes measured noise because",
     ["Ground currents from two points meet and their potential difference adds a spurious voltage",
      "The ground wire is too thin", "The decoupling caps are too small", "The supply is not isolated"], 0, "power"),

    # ------------------------------------------------------------ sensors
    ("sens/lsb", "A 12-bit ADC with a 3.0 V reference has an LSB of",
     ["3.0 V", "0.73 mV", "1.22 mV", "0.25 V"], 1, "sensors"),
    # Choice 1 ("The resolution in bits") restated the gold, so two options were
    # correct - averaging does raise ENOB. The duplicate is replaced by a claim
    # that is actually false.
    ("sens/oversampling", "Averaging 16 samples of a noisy ADC reading improves",
     ["It increases the ADC's nominal bit depth", "Effective resolution (ENOB) but not the nominal bit depth",
      "Both the nominal bit depth and the LSB", "Nothing"], 1, "sensors"),
    ("sens/thermocouple", "For measuring up to 1200 degrees C, the appropriate sensor is a",
     ["NTC thermistor", "DHT22", "K-type thermocouple with a MAX31855 amplifier", "LM75"], 2, "sensors"),
    ("sens/dht-accuracy", "How does the DHT22 compare with the DHT11 in temperature accuracy?",
     ["DHT22 is less accurate", "DHT22 is more accurate, about +/-0.2 C versus +/-0.5 C",
      "They are identical", "DHT22 cannot measure temperature"], 1, "sensors"),
    # Topology is now stated, because "what does the ADC read when it is dark" has
    # two defensible answers depending on which side of the divider the LDR sits.
    # 5V -> LDR -> A0 -> 10k -> GND is the canonical Arduino wiring, and in the
    # dark the LDR's resistance rises, pulling A0 up to VCC.
    ("sens/ldr-divider", "An LDR divider is wired 5V -> LDR -> ADC pin -> 10k -> GND. "
     "What does the ADC read when it is very dark?",
     ["Near VCC (LDR resistance high)", "Near 0 V (LDR resistance low)", "Exactly VCC/2", "It reads noise"], 0, "sensors"),

    # ------------------------------------------------------ debugging / practice
    ("dbg/logic-analyser", "For debugging a suspected I2C or SPI fault, a logic analyser is preferred to an oscilloscope because",
     ["It is cheaper and decodes the protocol in software",
      "It has higher bandwidth", "It measures voltage more accurately", "It can capture longer records"], 0, "debug"),
    ("dbg/floating-pin", "An unconnected input floating high or low at random is best fixed by",
     ["A pull-up or pull-down resistor", "A longer wire", "A higher sample rate", "Using analogRead instead"], 0, "debug"),
    ("dbg/i2c-pullup-missing", "An I2C bus reads all zeros on SDA. The most common cause is",
     ["The master is clocking too fast", "Missing or too-strong pull-up resistors leaving the line low",
      "The slave address is wrong", "The clock line is inverted"], 1, "debug"),
    ("dbg/brownout", "A device resets randomly under motor load. The most likely cause is",
     ["Flash wear", "Supply droop when the motor draws current, tripping the brown-out detector",
      "An I2C bus fault", "Excessive baud rate"], 1, "debug"),
    ("dbg/pwm-freq", "Why should PWM frequency be kept above about 20 kHz when driving a motor?",
     ["To reduce CPU load", "To push it above human hearing and above the amplifier's passband",
      "To increase torque", "To reduce ripple current"], 1, "debug"),
    ("dbg/millis-rollover", "Why must elapsed time be computed as (uint32_t)(now - last) rather than now - last?",
     ["It is faster", "Unsigned subtraction wraps correctly at the 49.7-day millis() rollover",
      "It avoids floating point", "millis() returns signed values"], 1, "debug"),
]


def electronics():
    return [{"id": f"el/{i}", "source": "hand-authored/embedded-practice",
             "category": "electronics_mc", "lang": "en",
             "question": q, "choices": c, "answer": a, "subject": s}
            for i, q, c, a, s in ELECTRONICS]


# --------------------------------------------------------------------------- C++
# Minimal Arduino/ESP32/Linux-GPIO surface. Enough for the logic that actually
# breaks; not a simulator (there is no AVR cross-compiler here to produce a
# sketch we could flash anyway).
MOCK_HEADER = r"""
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <cstdio>

typedef uint8_t byte;
#define HIGH 1
#define LOW 0
#define INPUT 0
#define OUTPUT 1
#define INPUT_PULLUP 2
#define OUTPUT_OPEN_DRAIN 3
#define LED_BUILTIN 13
#define DEC 10
#define HEX 16
#define A0 14
#define PI 3.14159265358979

// GPIO state, so a test can assert what the sketch actually drove the pin to.
static int g_pinmode[64];
static int g_digital_level[64];
static uint32_t g_millis = 0;
static uint32_t g_micros = 0;
static long g_analog_in[64];

static void pinMode(int pin, int mode) { g_pinmode[pin & 63] = mode; }
static void digitalWrite(int pin, int val) { g_digital_level[pin & 63] = val; }
static int  digitalRead(int pin) { return g_digital_level[pin & 63]; }
static int  analogRead(int ch) { return (int) g_analog_in[ch & 63]; }
static uint32_t millis(void) { return g_millis; }
static uint32_t micros(void) { return g_micros; }
static void delay(uint32_t ms) { g_millis += ms; g_micros += ms * 1000ULL; }
static void delayMicroseconds(uint32_t us) { g_micros += us; }
static void advanceTime(uint32_t ms) { g_millis += ms; g_micros += ms * 1000ULL; }
static int  pinLevel(int pin) { return g_digital_level[pin & 63]; }
static int  pinConfiguration(int pin) { return g_pinmode[pin & 63]; }
static void setAnalog(int ch, long v) { g_analog_in[ch & 63] = v; }
static void seedTime(uint32_t ms) { g_millis = ms; g_micros = ms * 1000ULL; }
static int  readByteFromStream(int) { return -1; }
static void Serial2_begin(unsigned long) {}
static void Serial2_print(const char*) {}
static void Serial2_println(const char*) {}
static void Serial2_print(int) {}
static void Serial2_println(int) {}
static void Serial2_print(float) {}
static void Serial2_println(float) {}
static void Serial2_write(int) {}

// ESP32 / Pi surface commonly used in real firmware.
static int esp_sleep(uint64_t) { return 0; }
static void ledcSetup(int, int, int) {}
static void ledcAttachPin(int, int) {}
static void ledcWrite(int, int) {}
static int  random(int) { return 0; }
static long random(long) { return 0; }
static void shiftOut(int, int, int, int) {}
static void shiftIn(int, int, int, int&) {}
static uint16_t crc16(const uint8_t*, size_t);
"""


def embedded_cpp():
    T = [
        ("embc/millis-rollover",
         "Write a C++ function `bool elapsed(uint32_t now, uint32_t last, uint32_t interval)` "
         "that returns true when `interval` milliseconds have passed since `last`. It MUST "
         "remain correct across the 49.7-day millis() rollover, so a naive signed comparison "
         "is wrong. Do not write main().",
         "assert(elapsed(1000, 0, 100) == true);\n"
         "assert(elapsed(500, 0, 1000) == false);\n"
         "assert(elapsed(100, 0, 100) == true);\n"
         "assert(elapsed(99, 0, 100) == false);\n"
         "assert(elapsed(0x80000064u, 0u, 100) == true);\n"
         "seedTime(0xFFFFFFFFu - 50u); advanceTime(200);\n"
         "assert(elapsed(millis(), 0xFFFFFF00u, 100) == true);"),

        ("embc/debounce",
         "Write C++ code implementing a button debouncer. Define a `ButtonDebouncer` struct "
         "with an `update(uint32_t now, bool raw_pressed)` method returning the debounced "
         "pressed state (true only after the raw input has been continuously stable for "
         "DEBOUNCE_MS). DEBOUNCE_MS must be defined as 20. Do not write main().",
         "ButtonDebouncer b;\n"
         "assert(b.update(0, false) == false);\n"
         "assert(b.update(5, true) == false);\n"
         "assert(b.update(10, true) == false);\n"
         "assert(b.update(25, true) == true);\n"
         "assert(b.update(30, true) == true);\n"
         "assert(b.update(40, false) == true);\n"
         "assert(b.update(65, false) == false);"),

        ("embc/pwm-timing",
         "Write C++ `uint32_t highTimeMicros(uint32_t dutyPermille, uint32_t periodMicros)` "
         "returning the ON time in microseconds for a given duty cycle expressed in "
         "permilles (0..1000). A duty above 1000 must be clamped to 1000. The arguments are "
         "unsigned, so do not try to handle negative values. Do not write main().",
         "assert(highTimeMicros(500, 2000) == 1000);\n"
         "assert(highTimeMicros(0, 2000) == 0);\n"
         "assert(highTimeMicros(1000, 2000) == 2000);\n"
         "assert(highTimeMicros(1500, 2000) == 2000);\n"
         "assert(highTimeMicros(250, 16000) == 4000);\n"
         "assert(highTimeMicros(1, 1000) == 1);"),

        ("embc/moving-average",
         "Write C++ implementing a fixed-size ring buffer average sensor filter. Provide "
         "`struct MovingAvg` with `void push(int16_t)`, `int16_t value() const` and "
         "`size_t count() const`. It must hold the last 8 samples, discard the oldest when "
         "full, and return the integer mean of the samples held. Do not write main().",
         "MovingAvg a;\n"
         "assert(a.count() == 0);\n"
         "a.push(10); assert(a.count() == 1 && a.value() == 10);\n"
         "a.push(20); assert(a.count() == 2 && a.value() == 15);\n"
         "a.push(30); assert(a.count() == 3 && a.value() == 20);\n"
         "for (int i = 4; i <= 8; i++) a.push((int16_t)(i * 10));\n"
         "assert(a.count() == 8);\n"
         "assert(a.value() == 45);\n"
         "a.push(90);\n"
         "assert(a.count() == 8);\n"
         "assert(a.value() == 55);"),

        ("embc/median-filter",
         "Write C++ `int16_t median3(int16_t a, int16_t b, int16_t c)` returning the median "
         "of three samples using a constant number of comparisons and no sorting library. "
         "Do not write main().",
         "assert(median3(3, 1, 2) == 2);\n"
         "assert(median3(1, 2, 3) == 2);\n"
         "assert(median3(3, 2, 1) == 2);\n"
         "assert(median3(5, 5, 5) == 5);\n"
         "assert(median3(-2, -9, -5) == -5);\n"
         "assert(median3(1000, -1000, 0) == 0);"),

        ("embc/pid",
         "Write C++ for a positional PID controller. Define `struct PID` holding float kp, ki, "
         "kd, integral, last_error and last_time, with a method "
         "`float update(float setpoint, float measured, uint32_t now_ms)` returning the "
         "correction. The integral term must accumulate error times elapsed "
         "SECONDS and be clamped to +/-1000; the derivative term must be "
         "divided by the elapsed time in seconds, guarding against zero "
         "elapsed time. Do not write main().",
         "PID p{2.0f, 0.5f, 1.0f};\n"
         "seedTime(0);\n"
         "float u1 = p.update(100.0f, 90.0f, millis());\n"
         "assert(u1 > 0.0f);\n"
         "advanceTime(1000);\n"
         "float u2 = p.update(100.0f, 100.0f, millis());\n"
         "assert(u2 < u1);\n"
         "for (int i = 0; i < 10000; i++) { advanceTime(10); p.update(100.0f, 0.0f, millis()); }\n"
         "assert(p.integral <= 1000.0f && p.integral >= -1000.0f);\n"
         "PID q{0.0f, 1.0f, 0.0f};\n"
         "seedTime(0);\n"
         "q.update(0.0f, 10.0f, millis());\n"
         "advanceTime(100);\n"
         "q.update(0.0f, 10.0f, millis());\n"
         "assert(q.integral > -1.5f && q.integral < -0.5f);\n"
         "assert(std::isfinite(q.update(0.0f, 5.0f, millis())));"),

        ("embc/adc-plausibility",
         "Write C++ `bool sensorReadingSane(int rawAdc, float volts, float expected, float tolerance)` "
         "returning true only when the reading is inside 0..1023, the voltage is non-negative, "
         "and the voltage is within `tolerance` of `expected`. Do not write main().",
         "assert(sensorReadingSane(512, 2.5f, 2.5f, 0.1f) == true);\n"
         "assert(sensorReadingSane(-1, 0.0f, 2.5f, 0.1f) == false);\n"
         "assert(sensorReadingSane(1024, 3.0f, 3.0f, 0.1f) == false);\n"
         "assert(sensorReadingSane(500, 2.4f, 2.5f, 0.05f) == false);\n"
         "assert(sensorReadingSane(0, 0.0f, 0.0f, 0.1f) == true);\n"
         "assert(sensorReadingSane(500, -1.0f, -1.0f, 0.1f) == false);"),

        ("embc/i2c-register-write",
         "Write C++ producing the byte sequence for an SMBus/I2C write of a 16-bit value to "
         "register 0x20 on device address 0x48. Provide "
         "`size_t encodeRegisterWrite(uint8_t devAddr, uint8_t reg, uint16_t value, uint8_t* out)` "
         "that writes device address, register high byte, register low byte, value high byte, "
         "value low byte and returns the 5 bytes written. Addresses and registers are passed "
         "already shifted into the standard 7-bit-to-8-bit write position; do not shift them "
         "again. Do not write main().",
         "uint8_t buf[8] = {0};\n"
         "size_t n = encodeRegisterWrite(0x90, 0x20, 0xBEEF, buf);\n"
         "assert(n == 5);\n"
         "assert(buf[0] == 0x90);\n"
         "assert(buf[1] == 0x00);\n"
         "assert(buf[2] == 0x20);\n"
         "assert(buf[3] == 0xBE);\n"
         "assert(buf[4] == 0xEF);"),

        ("embc/state-machine",
         "Write C++ for a non-blocking button state machine. Define `enum class BtnEvent { None, "
         "ShortPress, LongPress, DoublePress }` and `struct ButtonSM` with `BtnEvent "
         "update(uint32_t now, bool pressed, uint32_t long_ms, uint32_t dbl_ms)`. Semantics: "
         "press, then release in less than long_ms, yields ShortPress; holding for long_ms "
         "yields LongPress exactly once; if the button is pressed again within dbl_ms of the "
         "previous release, that press yields DoublePress instead of a second ShortPress. "
         "The function must never block. Do not write main().",
         "ButtonSM sm;\n"
         "seedTime(0);\n"
         "assert(sm.update(millis(), true, 1000, 250) == BtnEvent::None);\n"
         "advanceTime(1500);\n"
         "assert(sm.update(millis(), true, 1000, 250) == BtnEvent::LongPress);\n"
         "assert(sm.update(millis(), true, 1000, 250) == BtnEvent::None);\n"
         "advanceTime(200);\n"
         "assert(sm.update(millis(), false, 1000, 250) == BtnEvent::None);\n"
         "seedTime(10000);\n"
         "assert(sm.update(millis(), true, 1000, 250) == BtnEvent::None);\n"
         "advanceTime(100);\n"
         "assert(sm.update(millis(), false, 1000, 250) == BtnEvent::ShortPress);\n"
         "advanceTime(150);\n"
         "assert(sm.update(millis(), true, 1000, 250) == BtnEvent::DoublePress);\n"
         "advanceTime(50);\n"
         "assert(sm.update(millis(), false, 1000, 250) == BtnEvent::ShortPress);"),

        ("embc/watchdog-refresh",
         "Write C++ `bool refreshWatchdog(uint32_t now, uint32_t last, uint32_t timeout_ms)` "
         "returning true when the watchdog should be kicked, at most once per timeout_ms, "
         "and always true when the timeout has been exceeded. It must be correct across the "
         "32-bit millis() rollover. Do not write main().",
         "assert(refreshWatchdog(0, 0, 1000) == false);\n"
         "assert(refreshWatchdog(500, 0, 1000) == false);\n"
         "assert(refreshWatchdog(1200, 0, 1000) == true);\n"
         "assert(refreshWatchdog(1300, 1200, 1000) == false);\n"
         "assert(refreshWatchdog(0x80000000u, 0u, 1000) == true);\n"
         "assert(refreshWatchdog(0xFFFFFFFFu - 10u, 0xFFFF0000u, 500) == true);"),

        ("embc/serial-parse",
         "Write C++ `bool parseCsvLine(const char* line, float* out, size_t maxOut)` splitting a "
         "comma-separated line of floats into `out`, returning false on an empty line, a "
         "non-numeric field, or more fields than `maxOut`. Use strtof so it is not locale-blind. "
         "Do not write main().",
         "float v[4] = {0};\n"
         "assert(parseCsvLine(\"1.5,2.5,3\", v, 4) == true);\n"
         "assert(v[0] == 1.5f && v[1] == 2.5f && v[2] == 3.0f);\n"
         "assert(parseCsvLine(\"\", v, 4) == false);\n"
         "assert(parseCsvLine(\"1,abc\", v, 4) == false);\n"
         "assert(parseCsvLine(\"1;2\", v, 4) == false);\n"
         "assert(parseCsvLine(\"1 2\", v, 4) == false);\n"
         "assert(parseCsvLine(\"1,2,3,4,5\", v, 4) == false);\n"
         "assert(parseCsvLine(\"-40.5\", v, 4) == true && v[0] == -40.5f);"),

        ("embc/crc16-modbus",
         "Write C++ `uint16_t crc16Modbus(const uint8_t* data, size_t len)` computing the "
         "standard Modbus RTU CRC-16: initial value 0xFFFF, reflected polynomial 0xA001, no "
         "final XOR. Do not write main().",
         "uint8_t d1[] = {'1','2','3','4','5','6','7','8','9'};\n"
         "assert(crc16Modbus(d1, sizeof(d1)) == 0x4B37);\n"
         "uint8_t d2[] = {0x01, 0x04, 0x02, 0xFF, 0xFF};\n"
         "assert(crc16Modbus(d2, sizeof(d2)) == 0x80B8);\n"
         "uint8_t d3[] = {0x01, 0x03, 0x00, 0x00, 0x00};\n"
         "assert(crc16Modbus(d3, sizeof(d3)) == 0x8419);\n"
         "uint8_t d4[] = {0xAA, 0x55};\n"
         "assert(crc16Modbus(d4, sizeof(d4)) == 0x2FBF);"),
    ]
    return [{"id": i, "source": "hand-authored/embedded-cpp", "category": "embedded_cpp",
             "lang": "cpp", "prompt": p, "test": t, "header": MOCK_HEADER}
            for i, p, t in T]


if __name__ == "__main__":
    e = electronics()
    c = embedded_cpp()
    # De-bias in the build path, not as a separate manual step nobody calls.
    debias_mc.rebalance(e)
    hist, best = debias_mc.report(e)
    print(f"       (electronics gold positions {hist}, best-constant baseline {best:.1%})")
    for nm, rows in (("electronics", e), ("embedded_cpp", c)):
        with open(os.path.join(OUT, nm + ".json"), "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"  {nm:<20} {len(rows):>4} tasks -> {OUT}/{nm}.json")
    subs = {}
    for r in e:
        subs[r["subject"]] = subs.get(r["subject"], 0) + 1
    print("  electronics by subject:", ", ".join(f"{k}={v}" for k, v in sorted(subs.items())))