
# ensemble th escores together

chip <- read.csv('submit/chippys.txt', stringsAsFactors = F)
chip90gain <- read.csv('submit/chippy_gain90.txt', stringsAsFactors = F)
chip99gain <- read.csv('submit/chippy_gain99.txt', stringsAsFactors = F)
chip90freq <- read.csv('submit/chippy_freq90.txt', stringsAsFactors = F)
chip99freq <- read.csv('submit/chippy_freq99.txt', stringsAsFactors = F)


suball = Reduce(function(x,y) merge(x,y, by="id"), list(chip, chip90gain, chip99gain, chip90freq, chip99freq))


# ensemble with the median score
tmp$agg <- apply(suball[,-1], 1, median)
ens.median <- tmp[,c('id','agg')]
names(ens.median) <- c('id','loss')
write.table(ens.median, 'submit/chip_ens5_med.txt', quote=F, row.names=F, sep=',')

# ensemble with the average score
tmp$agg <- apply(suball[,-1], 1, mean, na.rm=T)
ens.median <- tmp[,c('id','agg')]
names(ens.median) <- c('id','loss')
write.table(ens.median, 'submit/chip_ens5_avg.txt', quote=F, row.names=F, sep=',')
