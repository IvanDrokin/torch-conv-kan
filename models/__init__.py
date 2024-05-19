from .baselines.conv_kaln_baseline import SimpleConvKALN, EightSimpleConvKALN
from .baselines.conv_kan_baseline import SimpleConvKAN, EightSimpleConvKAN
from .baselines.fast_conv_kan_baseline import SimpleFastConvKAN, EightSimpleFastConvKAN
from .baselines.conv_baseline import SimpleConv, EightSimpleConv
from .baselines.conv_kacn_baseline import SimpleConvKACN, EightSimpleConvKACN

from .reskanet import reskanet_18x32p, reskacnet_18x32p, fast_reskanet_18x32p, reskalnet_18x32p, reskalnet_18x64p, \
    reskalnet_50x64p, moe_reskalnet_50x64p, reskalnet_101x64p, moe_reskalnet_101x64p, \
    reskalnet_152x64p, moe_reskalnet_152x64p, moe_reskalnet_18x64p
from .ukanet import ukanet_18, ukalnet_18, fast_ukanet_18, ukacnet_18
