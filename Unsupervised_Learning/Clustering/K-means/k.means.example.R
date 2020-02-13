### Association of lncRNAs and mRNAs: K-means
### February 13th 2020
### Libraries:
library(tidyverse)
library(magrittr)
library(reshape2)
options(stringsAsFactors = F)
setwd("/users/rg/ramador/D_me/RNA-seq/mRNA_lncRNA_patternanalysis")
source("/users/rg/ramador/Scripts/R-functions/Regeneration/color.palette.R")
library(factoextra)
library(NbClust)

### input data
input <- readRDS("Results//association.mRNA.lncRNA.input.RDS")

input$pairs <- paste(input$Gene_Name,input$partnerRNA_gene, sep="_" )
input %<>% select(pairs,L3:Regeneration.25h,
                 L3_mRNA:mRNA_Regeneration.25h) %>% as.data.frame
rownames(input) <- input$pairs; input$pairs <- NULL

#### --- 1) k-means analysis: 

# Log transformation:
log_input <- log10(input+0.01)

## Observe the number of cluster: 
fviz_nbclust(log_input, kmeans, method = "wss")

### K-means: 
set.seed(123)
innitial.kms.log <- kmeans(log_input, 7, nstart = 25)
innitial.kms.log$size

plot_kmeans <- innitial.kms.log$centers


#### --- 2) Analyze the cluster: 

## pdf(file = "Plots//association.cluster.analysis.pdf", paper = "a4r",
##     width = 0, height = 0)

fviz_cluster(innitial.kms.log, data = log_input)+
    ggtitle("Analysis of clusters: association lncRNA-PCG")+
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
dev.off()


#### --- 3) Plot cluster: 

innitial.kms.log$size

## pdf(file = "Plots//association.clusters.lncRNA.mRNA.pdf", paper = "a4r",
##     width = 0, height = 0)

for(i in 1:nrow(plot_kmeans)){

    cat(i, "\n")
    
    tmp <- melt(plot_kmeans[i,])
    tmp$variable <- row.names(tmp)
    rownames(tmp) <- 1:nrow(tmp)
    tmp$variable <- as.character(tmp$variable)
    tmp$variable[c(2:7,9:14)] %<>% 
        strsplit(., split=".", fixed=TRUE) %>% lapply(., function(x){y <- x[2]}) %>%
        unlist
    tmp$variable[8] <- "L3"
    tmp$variable <- as.factor(tmp$variable)
    tmp$variable <- factor(tmp$variable,
                           levels = levels(tmp$variable)[c(4,1:3)])

    #values:
    gene_value <- tmp[1,1]
    partner_value <- tmp[8,1]

    tmp <- tmp %>% add_row(.,variable="L3", value=gene_value, .after = 4 ) %>%
        add_row(., variable="L3", value=partner_value, .after = 12)

    tmp$Treatment <- rep(c(rep("Control", 4), rep("Regeneration",4)),)
    tmp$gene_type <- c(rep("lncRNA",8), rep("PCG", 8))


    p <- ggplot(data=tmp, aes(x=variable, y=value, group=Treatment))+
        geom_line(aes(linetype=Treatment, color=Treatment))+
        facet_wrap(~gene_type, scales = "free")+geom_point(aes(color=Treatment))+
        xlab("")+ylab("log10(TPM+0.01)")+theme_light()+
        theme(legend.position = "bottom", strip.text = element_text(size=11))+
        labs(color="")+
        labs(linetype="")+
        scale_color_manual(values = c(control_sample, regene_sample))

    plot(p)
    
}

dev.off()




