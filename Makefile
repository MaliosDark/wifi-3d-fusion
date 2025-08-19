.PHONY: install run realtime

install:
	bash scripts/install_all.sh

run:
	bash scripts/run_realtime.sh --source esp32

realtime:
	bash scripts/run_realtime.sh --source $(SRC)
