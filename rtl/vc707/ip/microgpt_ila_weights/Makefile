PROJECT := microgpt_ila_weights

all:
	vivado -mode batch -source tcl/run.tcl

clean:
	rm -rf ip $(PROJECT).* component.xml vivado*.jou vivado*.log vivado*.str xgui .Xil
