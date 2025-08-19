# scripts/setup_monitor.sh
#!/usr/bin/env bash
set -euo pipefail

# 1) Detecta interfaz Wi-Fi si no se pasa IFACE
IFACE="${1:-}"
if [[ -z "${IFACE}" ]]; then
  IFACE=$(ls /sys/class/net | grep -E 'wl|wlan' | head -n1 || true)
fi
if [[ -z "${IFACE}" ]]; then
  echo "No Wi-Fi interface found."; exit 1
fi

echo "[*] Using IFACE=${IFACE}"

# 2) Libera la interfaz de NetworkManager/WPA
sudo nmcli dev set "${IFACE}" managed no || true
sudo systemctl stop wpa_supplicant 2>/dev/null || true

# 3) Sube/Down limpio y elimina mon0 si existe
sudo ip link set "${IFACE}" down || true
sudo iw dev mon0 del 2>/dev/null || true

# 4) Resuelve el PHY de la interfaz
PHY_LINK="$(readlink -f "/sys/class/net/${IFACE}/phy80211" 2>/dev/null || true)"
if [[ -z "${PHY_LINK}" || ! -e "${PHY_LINK}" ]]; then
  # fallback via iw
  PHY=$(iw dev | awk -v ifc="${IFACE}" '
    $1=="Interface" && $2==ifc {print phy; exit}
    $1~"^phy#"[0-9]+ {phy=$1}
  ')
else
  PHY="$(basename "${PHY_LINK}")"
fi

if [[ -z "${PHY}" ]]; then
  echo "Could not resolve PHY for ${IFACE}"; exit 1
fi
echo "[*] Using ${PHY}"

# 5) Crea interfaz monitor y súbela en canal 6/HT20
sudo iw phy "${PHY}" interface add mon0 type monitor
sudo ip link set mon0 up
sudo iw dev mon0 set channel 6 HT20 || sudo iw dev mon0 set channel 1 HT20

# 6) Verifica y prueba tráfico
iw dev mon0 info
echo "[*] Testing traffic on mon0 (20 frames)…"
sudo tcpdump -I -i mon0 -s 0 -vv -c 20
