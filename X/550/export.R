load("nets_yearly.RData")
library(reticulate)
# import numpy
np = import("numpy")

for(i in 1:11){
  filename = paste('Y',
                   i, '.npy', sep='')
  np$save(filename,r_to_py(nets_thre[[i]]))
}

for(i in 1:11){
  filename = paste('name',
                   i, '.npy', sep='')
  np$save(filename,r_to_py(colnames(nets_thre[[i]])))
}

features_sub = NULL
for(t in 1:11){
  nodes = colnames(nets_thre[[t]])
  for(i in 1:length(nodes)){
    features_sub = rbind(features_sub, network_features[network_features['screen_name'] == nodes[i], ])
  }
}
np$save('pi.npy',r_to_py(features_sub['party_id'][[1]]))

np$save('real_name.npy',r_to_py(network_features['real_name'][[1]]))
np$save('handle.npy',r_to_py(network_features['screen_name'][[1]]))
np$save('party_id.npy',r_to_py(network_features['party_id'][[1]]))

