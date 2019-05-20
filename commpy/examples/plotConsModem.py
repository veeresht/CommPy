# Authors: Youness Akourim <akourim97@gmail.com>
# License: BSD 3-Clause

from commpy.modulation import PSKModem, QAMModem

# =============================================================================
# Example constellation plot of Modem
# =============================================================================

# PSK corresponding to PSKModem for 4 bits
psk = PSKModem(16)
psk.plotCons()

# QAM corresponding to QAMModem for 2bits
Qam = QAMModem(4)
Qam.plotCons()