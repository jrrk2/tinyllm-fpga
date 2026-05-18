PROJECT := microgpt_ila_axi

all:
	vivado -mode batch -source tcl/run.tcl

clean:
	rm -rf ip $(PROJECT).* component.xml vivado*.jou vivado*.log vivado*.str xgui .Xil
