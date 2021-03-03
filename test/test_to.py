import snap

# UGraph1 = snap.GenRndPowerLaw (9, 10)
# for NI in UGraph1.Nodes():
#     print("node: %d, out-degree %d, in-degree %d" % (NI.GetId(), NI.GetOutDeg(), NI.GetInDeg()))

UGraph2 = snap.GenRndPowerLaw (5, 2, False)
for NI in UGraph2.Nodes():
    print("node: %d, out-degree %d, in-degree %d" % (NI.GetId(), NI.GetOutDeg(), NI.GetInDeg()))