library(igraph)
library(ggplot2)

# application plotting 
# density plots
# STANDARD
# --> Intersection weighting:
nets = list()
for(i in 1:11){
  filename = paste('../yearly/aggregated_network_hashtag_intersection_year_',
                   i-1+2, '.csv', sep='')
  nets[[i]] = read.csv(filename, row.names = 1, header = TRUE)
}

# polarized
# nets = list()
# for(i in 1:11){
#   filename = paste('../../congressional_hashtag_networks/congressional_hashtag_networks_data/hashtag_networks_data_updated_Nov/year_aggregated/hashtag_intersection_weighted/aggregated_network_hashtag_intersection_polarizing_hashtags_year_',
#                    i-1+2, '.csv', sep='')
#   nets[[i]] = read.csv(filename, row.names = 1, header = TRUE)
# }

network_features = read.csv('../yearly/network_features.csv')

nets_reduced = list()
for (i in 1:length(nets)){
  nets_reduced[[i]] = nets[[i]]
}

nets_thre = list()
TT = length(nets_reduced)
for(t in 1:TT){
  edges_counts = nets_reduced[[t]][lower.tri(nets_reduced[[t]])] 
  thre = mean(edges_counts)
  nets_thre[[t]] = 1*(nets_reduced[[t]]>thre)
}

save(nets_thre, network_features, file = "nets_yearly.RData")

