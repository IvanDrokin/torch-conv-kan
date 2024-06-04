from .baselines.conv_kaln_baseline import SimpleConvKALN, EightSimpleConvKALN
from .baselines.conv_kan_baseline import SimpleConvKAN, EightSimpleConvKAN
from .baselines.fast_conv_kan_baseline import SimpleFastConvKAN, EightSimpleFastConvKAN
from .baselines.conv_baseline import SimpleConv, EightSimpleConv
from .baselines.conv_kacn_baseline import SimpleConvKACN, EightSimpleConvKACN
from .baselines.conv_kagn_baseline import SimpleConvKAGN, EightSimpleConvKAGN
from .baselines.conv_wavkan_baseline import SimpleConvWavKAN, EightSimpleConvWavKAN

from .reskanet import reskanet_18x32p, reskacnet_18x32p, fast_reskanet_18x32p, reskalnet_18x32p, reskalnet_18x64p, \
    reskalnet_50x64p, moe_reskalnet_50x64p, reskalnet_101x64p, moe_reskalnet_101x64p, \
    reskalnet_152x64p, moe_reskalnet_152x64p, moe_reskalnet_18x64p, reskalnet_101x32p, \
    reskalnet_152x32p, reskagnet_101x64p, reskagnet_18x32p, reskagnet50, reskagnet18
from .densekanet import densekanet121, densekalnet121, densekacnet121, densekagnet121, fast_densekanet121
from .densekanet import densekalnet161, densekalnet169, densekalnet201
from .densekanet import tiny_densekanet, tiny_densekalnet, tiny_densekacnet, tiny_fast_densekanet, tiny_densekagnet
from .vggkan import fast_vggkan, vggkan, vggkaln, vggkacn, vggkagn, moe_vggkagn, vgg_wav_kan

from .ukanet import ukanet_18, ukalnet_18, fast_ukanet_18, ukacnet_18, ukagnet_18
from .u2kanet import u2kagnet, u2kacnet, u2kalnet, u2kanet, fast_u2kanet
from .u2kanet import u2kagnet_small, u2kacnet_small, u2kalnet_small, u2kanet_small, fast_u2kanet_small
