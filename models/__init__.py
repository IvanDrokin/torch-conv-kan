from .automodel import AutoKAN, TinyAutoKAGN
from .baselines.conv_baseline import SimpleConv, EightSimpleConv
from .baselines.conv_kacn_baseline import SimpleConvKACN, EightSimpleConvKACN
from .baselines.conv_kagn_baseline import SimpleConvKAGN, EightSimpleConvKAGN
from .baselines.conv_kaln_baseline import SimpleConvKALN, EightSimpleConvKALN
from .baselines.conv_kan_baseline import SimpleConvKAN, EightSimpleConvKAN
from .baselines.conv_moe_kagn_bn_baseline import EightSimpleMoEConvKAGNBN, SimpleMoEConvKAGNBN
from .baselines.conv_wavkan_baseline import SimpleConvWavKAN, EightSimpleConvWavKAN
from .baselines.fast_conv_kan_baseline import SimpleFastConvKAN, EightSimpleFastConvKAN
from .densekanet import densekalnet161, densekalnet169, densekalnet201, densekagnet161, densekagnet169, densekagnet201
from .densekanet import densekanet121, densekalnet121, densekacnet121, densekagnet121, fast_densekanet121
from .densekanet import tiny_densekagnet_bn, densekagnet161bn, densekagnet169bn, densekagnet201bn, densekagnet121bn
from .densekanet import tiny_densekagnet_moebn, densekagnet161moebn, densekagnet169moebn, densekagnet201moebn, \
    densekagnet121moebn
from .densekanet import tiny_densekanet, tiny_densekalnet, tiny_densekacnet, tiny_fast_densekanet, tiny_densekagnet
from .reskanet import reskagnetbn_18x32p, reskagnetbn_moe_18x32p, reskagnet_34x32p, reskagnetbn_34x32p, \
    reskagnetbn_moe_34x32p
from .reskanet import reskanet_18x32p, reskacnet_18x32p, fast_reskanet_18x32p, reskalnet_18x32p, reskalnet_18x64p, \
    reskalnet_50x64p, moe_reskalnet_50x64p, reskalnet_101x64p, moe_reskalnet_101x64p, \
    reskalnet_152x64p, moe_reskalnet_152x64p, moe_reskalnet_18x64p, reskalnet_101x32p, \
    reskalnet_152x32p, reskagnet_101x64p, reskagnet_18x32p, reskagnet50, reskagnet18, reskagnet101, reskagnet152, \
    reskagnet_bn50
from .u2kanet import u2kagnet, u2kacnet, u2kalnet, u2kanet, fast_u2kanet, u2net, u2net_small, u2kagnet_bn
from .u2kanet import u2kagnet_small, u2kacnet_small, u2kalnet_small, u2kanet_small, fast_u2kanet_small, \
    u2kagnet_bn_small
from .ukanet import ukanet_18, ukalnet_18, fast_ukanet_18, ukacnet_18, ukagnet_18, ukagnetnb_18
from .vggkan import fast_vggkan, vggkan, vggkaln, vggkacn, vggkagn, moe_vggkagn, vgg_wav_kan, vggkagn_bn, moe_vggkagn_bn
from .ukanet import UKAGNet

from .vggkan import VGGKAGN_BN, VGGKAGN
