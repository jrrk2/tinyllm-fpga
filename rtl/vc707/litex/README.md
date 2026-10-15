# tinyllm on a LiteX VC707 SoC

SmolLM2-360M, block-floating-point, running in hardware on a Xilinx
**VC707** board.  The tinyllm engine is a peripheral of a bare-metal LiteX
SoC (VexRiscv, 1 GiB DDR3 via LiteDRAM, LiteEth, LiteSDCard).  At power-up
the firmware reads the model from an SD card into DDR3; you then type a
prompt on the serial console and the FPGA writes the continuation.

| | |
|---|---|
| Model | SmolLM2-360M: D=960, 32 layers, 15/5 heads, FFN 2560, vocab 49152 |
| Context | prompt up to 24 tokens, 8 generated tokens (32-token KV cache) |
| Weights | 843 MB packed BFP image, loaded from SD in about 2.5 minutes |
| Clocks | SoC 100 MHz, engine 40 MHz |
| Verified | 100 of 100 restarts give the golden token sequence |

Everything below is run from `rtl/vc707/`.

## Install from a release

You need:

- a VC707 board, its USB-JTAG cable and USB-UART cable
- a full-size SD card (or microSD in an adapter) of at least 1 GB
- openFPGALoader or Vivado Hardware Manager to program the FPGA
- a serial terminal (picocom, minicom, PuTTY)

### 1. Prepare the SD card

The card must be **FAT32**.  exFAT is not supported, and cards larger
than 32 GB come formatted as exFAT, so reformat those as FAT32.

Copy the five files from the release's SD-card bundle to the root of the
card:

| File | Contents |
|---|---|
| `WEIGHTS.BIN` | DDR3 weight image (843 MB) |
| `ROMS.BIN` | layer-norm gammas, final norm, default prompt |
| `VOCAB.BIN` | tokenizer table for the serial console |
| `FW.BIN` | the bare-metal firmware |
| `boot.json` | tells the LiteX BIOS where to load `FW.BIN` |

Check the copy against `SHA256SUMS`, then unmount the card and insert it
in the VC707 SD slot.

### 2. Program the FPGA

Over JTAG (lost at power-off):

```
openFPGALoader -c digilent --freq 15000000 tinyllm_vc707_litex.bit
```

or open the bitstream in Vivado Hardware Manager and press Program.

### 3. Talk to it

Open the USB-UART port at 115200 baud:

```
picocom -b 115200 --imap lfcrlf /dev/ttyUSB2    # the port number varies
```

The LiteX BIOS first tries a network boot from TFTP server
192.168.1.106:6969.  When nothing answers it boots `FW.BIN` from the SD
card.  The firmware then loads the model, which takes about 2.5 minutes
and shows a progress bar.  When the `>` prompt appears, type some text
and press Enter.

Console commands (one letter + Enter):

| Key | Action |
|---|---|
| `?` | menu |
| `v` | bitstream build version |
| `s` | engine status and model shape |
| `t` | rerun the stored prompt and print the tokens |
| `r` | pulse engine restart |
| `c` | weight hash |
| `l` | reload the model from SD |
| `d` | list the SD card |
| `e` | Ethernet status |
| `R` | reboot to the BIOS |

Any input of two or more characters is treated as a prompt.

## Ethernet control (optional)

The board answers on **192.168.1.50**, UDP port 19783, with MAC
10:e2:d5:54:4c:4d.  Both addresses are fixed in the bitstream.  The host
client is built with:

```
make host/bfp_client
```

```
# run the stored prompt and decode the result
MGRT_PREFIX=lbfp_full_ host/bfp_client -p 192.168.1.50:19783 tokens

# upload a weight image without the SD card (about 2 minutes);
# 1 MiB block checksums are compared and only bad blocks are resent
host/bfp_client -p 192.168.1.50:19783 -r 20 blast lbfp_full_DDR3.bin

# check the DDR3 image against a local file
host/bfp_client -p 192.168.1.50:19783 cksum lbfp_full_DDR3.bin

# reload gammas + the default prompt from a generated/ directory
MGRT_PREFIX=lbfp_full_ host/bfp_client -p 192.168.1.50:19783 load-roms ../generated
```

A prompt typed on the console replaces the stored prompt.  Run
`load-roms` to restore the default prompt before comparing with the
golden tokens `346 253 655 28 281 253 1666 1869 2025 28 665 4161`.

## Build from source

Prerequisites: Vivado 2020.1, the
[xc7-bitstream-tools](https://github.com/jrrk2/xc7-bitstream-tools)
checkout with its LiteX venv (`LITEX_TOOLS_DIR`, default
`~/xc7-bitstream-tools`), a RISC-V GCC, and the generated model files in
`rtl/generated/` (`lbfp_full_*`).

```
make litex-engine                          # SoC + engine bitstream (~1.5 h)
make litex-fw                              # firmware -> litex/fw/tinyllm_fw.bin
make litex-sd-bundle SD_OUT=/media/sdcard  # write the five SD files
make litex-engine-load                     # JTAG-program build/litex/gateware/xilinx_vc707.bit
```

For firmware development, `make litex-fw-deploy` puts `FW.BIN` and
`boot.json` in the per-MAC TFTP directory
(`~/tftp-vc707/10:e2:d5:54:4c:4d`), and the BIOS netboots it in place of
the SD copy.

## Files

| Path | What |
|---|---|
| `tinyllm_vc707.py` | the SoC: board base + engine, MAC/IP, boot order |
| `tinyllm.py` | the engine as a LiteX module (CSRs, 512-bit AXI read master into LiteDRAM) |
| `fw/` | bare-metal firmware: console, tokenizer, SD loader, UDP protocol |
| `sd/make_sd_bundle.py` | builds the SD-card files |
