"""
Script genere automatiquement par le simulateur Dobot.
Programme : cycle_pick_place

Prerequis : DobotDllType.py et les DLL Dobot dans le meme dossier.
ATTENTION : degagez la zone de travail avant de lancer ce script.
Outil monte : ventouse
"""

import DobotDllType as dType

api = dType.load()
etat = dType.ConnectDobot(api, "", 115200)[0]
if etat != dType.DobotConnect.DobotConnect_NoError:
    raise SystemExit("connexion au Dobot impossible : %s" % etat)

dType.SetQueuedCmdClear(api)
dType.SetHOMEParams(api, 200, 0, 0, 0, isQueued=1)
dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued=1)
dType.SetPTPCoordinateParams(api, 200, 200, 200, 200, isQueued=1)
dType.SetPTPCommonParams(api, 100, 100, isQueued=1)

dernier = 0

# repli
dType.SetPTPJointParams(api, 90.00, 90.00, 90.00, 90.00, 300.00, 300.00, 300.00, 300.00, isQueued=1)
dernier = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJANGLEMode, 0.00, 15.00, 30.00, 0.00, isQueued=1)[0]
# au-dessus de la piece
dernier = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, 220.00, -90.00, 60.00, 0.00, isQueued=1)[0]
# descente
dType.SetPTPCoordinateParams(api, 120.00, 120.00, 500.00, 500.00, isQueued=1)
dernier = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 220.00, -90.00, 25.00, 0.00, isQueued=1)[0]
# ventouse ON
dernier = dType.SetEndEffectorSuctionCup(api, 1, 1, isQueued=1)[0]
dernier = dType.SetWAITCmd(api, 400, isQueued=1)[0]
# remontee
dernier = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 220.00, -90.00, 110.00, 0.00, isQueued=1)[0]
# transfert
dernier = dType.SetARCCmd(api, [265.00, 0.00, 130.00, 0.00], [220.00, 90.00, 110.00, 45.00], isQueued=1)[0]
# descente
dernier = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 220.00, 90.00, 25.00, 45.00, isQueued=1)[0]
# ventouse OFF
dernier = dType.SetEndEffectorSuctionCup(api, 1, 0, isQueued=1)[0]
# retour
dType.SetPTPJointParams(api, 90.00, 90.00, 90.00, 90.00, 300.00, 300.00, 300.00, 300.00, isQueued=1)
dernier = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJANGLEMode, 0.00, 15.00, 30.00, 0.00, isQueued=1)[0]

dType.SetQueuedCmdStartExec(api)
while dernier > dType.GetQueuedCmdCurrentIndex(api)[0]:
    dType.dSleep(100)
dType.SetQueuedCmdStopExec(api)
dType.DisconnectDobot(api)
