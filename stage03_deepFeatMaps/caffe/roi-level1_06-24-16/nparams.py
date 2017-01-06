import sys
sys.path.append("/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/external/caffe/python")
import caffe
args = sys.argv
net = caffe.Net(args[1], caffe.TEST)

layercount = 0
for layer in net.layers:
    layercount += 1

# need to print layercount & paramcount & RF & in size & out size, we can get prop time manually

print "# Layers (Blobs): ", layercount

reallayercount = 0
paramscount = 0
for k, v in net.params.items():
#    print (k, v[0].data.shape, v[1].data.shape)
    reallayercount += 1

    _mult = 1
    for i in range(len(v[0].data.shape)):
        _mult *= v[0].data.shape[i]
    
    paramscount += _mult
    '''
    if len(v[0].data.shape) < 4:
        paramscount += v[0].data.shape[0] * v[0].data.shape[1]
    else:
        paramscount += v[0].data.shape[0] * v[0].data.shape[1] * v[0].data.shape[2] * v[0].data.shape[3]
    '''

print "# Params: ",  paramscount
print "# Real Layers: ", reallayercount
#print   [(k, v[0].data.shape[0] * v[0].data.shape[1] * v[0].data.shape[2] * v[0].data.shape[3], v[0].data.shape, v[1].data.shape) for k, v in net.params.items()[:-1]]
