import logic_gate as lg
gate = lg.LogicGate()

tests = [[0,0],[0,1],[1,0],[1,1]]

print('Test for AND gate')
for test in tests:
    y = gate.and_gate(test[0], test[1])
    print(f'{y}=AND({test[0]}, {test[1]})')

print('Test for NAND gate')
for test in tests:
    y = gate.nand_gate(test[0], test[1])
    print(f'{y}=NAND({test[0]}, {test[1]})')

print('Test for OR gate')
for test in tests:
    y = gate.or_gate(test[0], test[1])
    print(f'{y}=OR({test[0]}, {test[1]})')
    
print('Test for NOR gate')
for test in tests:
    y = gate.nor_gate(test[0], test[1])
    print(f'{y}=NOR({test[0]}, {test[1]})')
    
print('Test for XOR gate')
for test in tests:
    y = gate.xor_gate(test[0], test[1])
    print(f'{y}=XOR({test[0]}, {test[1]})')