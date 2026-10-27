suppressPackageStartupMessages({library(dplyr);library(tidyr);library(purrr);library(tibble);library(ggplot2);library(cowplot);library(patchwork);library(mcp);library(ggdist)})
root <- normalizePath('boundaryPilotV2_clinical_R')
source_file <- '/Users/matildacotton/Desktop/Masters_V2/EFOV_analysis_pipeline.Rmd'
file.copy(source_file,file.path(root,'original_EFOV_analysis_pipeline.Rmd'),overwrite=TRUE)
lines <- readLines(source_file)
chunk <- function(name) {
 start <- grep(paste0('^```\\{r ',name,'([,}])'),lines)[1]+1
 end <- which(seq_along(lines)>start & lines=='```')[1]-1
 paste(lines[start:end],collapse='\n')
}
for(n in c('config','helpers','model-functions','plot-helpers')) eval(parse(text=chunk(n)))
OUT_DIR <- root; FIG_DIR <- file.path(root,'figures'); TAB_DIR <- file.path(root,'tables'); CACHE_DIR <- file.path(root,'cache')
for(d in c(FIG_DIR,TAB_DIR,CACHE_DIR)) dir.create(d,recursive=TRUE,showWarnings=FALSE)
# cowplot version predates multiple guide-box positions in ggplot2.
get_legend <- function(plot) {
 g <- ggplotGrob(plot)
 candidates <- which(grepl('guide-box',g$layout$name))
 for(i in candidates) if(inherits(g$grobs[[i]],'gtable')) return(g$grobs[[i]])
 NULL
}
BOUNDARY_OFFSET <- 0
# Each standoff is an independent scenario; neither standoff nor noise is a patient.
ELECTRODES <- c('Unipolar','Bipolar','Omnipolar E5-E6-E9','Omnipolar E9-E10-E5')
COLORS <- setNames(c('#0072B2','#D55E00','#009E73','#8B5FBF'),ELECTRODES)
LOCATIONS <- 'Simulated slab'
pts <- read.csv(file.path(root,'trace_verified_points.csv'))
pts <- pts %>% filter(noiseRepeat==0) %>% mutate(patientID=factor('one_slab'), Location='Simulated slab',Electrode=factor(ifelse(kind=='uni','Unipolar',ifelse(kind=='bi','Bipolar',paste('Omnipolar',label))),levels=ELECTRODES),Dist=distanceInsideMm,PP=tracePP,PP_log=log(PP),PFreq=NA_real_,PFreq_log=NA_real_)
select_decay_model <- function(d,outcome) map_dfr(DECAY_MODELS,function(m){f<-lm(as.formula(paste(outcome,'~',decay_term(m))),data=filter(d,Dist>0));tibble(model=m,n=nobs(f),AIC=AIC(f),BIC=BIC(f))}) %>% mutate(dAIC=AIC-min(AIC),best=dAIC==0)
predict_decay <- function(fit,model,dist) { p<-predict(fit,newdata=data.frame(Dist=dist),interval='confidence');tibble(Dist=dist,value=exp(p[,1]),lo=exp(p[,2]),hi=exp(p[,3])) }
summaries <- list()
for(gap in sort(unique(pts$gapMm))) {
 message('Processing gap ',gap)
 FIG_DIR <- file.path(root,paste0('gap_',gap,'mm'),'figures');dir.create(FIG_DIR,recursive=TRUE,showWarnings=FALSE)
 raw_pre <- filter(pts,gapMm==gap,condition=='intact'); raw_post<-filter(pts,gapMm==gap,condition=='lesion');ds<-make_datasets(raw_post)
 specs <- ds$volt %>% distinct(Location,Electrode) %>% mutate(Metric='Voltage',outcome='PP_log', data=map(Electrode,~filter(ds$volt,Electrode==.x)))
 selection <- specs %>% mutate(sel=map2(data,outcome,select_decay_model)) %>% select(Location,Electrode,sel) %>% unnest(sel)
 specs <- specs %>% left_join(filter(selection,best) %>% select(Location,Electrode,model),by=c('Location','Electrode'))
 specs$fit_reml <- pmap(list(specs$data,specs$model),function(d,m) lm(as.formula(paste('PP_log ~',decay_term(m))),data=d))
 specs$fit_mcp <- pmap(list(specs$data,specs$model,specs$Electrode),function(d,m,e){message('Bayesian fit ',e);fit_piecewise(d,'PP_log',m,as.character(e))})
 primary <- list(specs=specs)
 bayes <- specs %>% mutate(t=map2(fit_mcp,model,~summarise_piecewise(.x,.y,'Voltage'))) %>% select(Location,Electrode,model,t) %>% unnest(t)
 efov <- bayes %>% filter(term=='cp_mm') %>% transmute(Location,Electrode,EFOV=mean)
 write.csv(selection,file.path(TAB_DIR,paste0('gap_',gap,'_model_selection.csv')),row.names=FALSE)
 write.csv(bayes,file.path(TAB_DIR,paste0('gap_',gap,'_bayesian_summary.csv')),row.names=FALSE)
 summaries[[as.character(gap)]] <- mutate(bayes,gapMm=gap)
 pred <- specs %>% mutate(pred=map2(fit_reml,model,~predict_decay(.x,.y,seq(min(ds$volt$Dist),15,length.out=200)))) %>% select(Location,Electrode,pred) %>% unnest(pred)
 pre <- raw_pre %>% in_roi() %>% filter(PP>DETECTION_LIMIT) %>% trim_upper('PP')
 p1 <- panel_observed(pre,'PP','Simulated slab','Observed (Pre-Ablation)',title=paste('Simulated slab - gap',gap,'mm'),ylim=c(0,8),ybreaks=seq(0,8,2))
 p2 <- panel_observed(ds$all,'PP','Simulated slab','Observed (Post-Ablation)',ylim=c(0,6),ybreaks=0:6,boundary=TRUE)
 p3 <- panel_model(ds$volt,pred,'PP','Simulated slab',ylim=c(0,2),ybreaks=seq(0,2,.5))
 legend <- get_legend(ggplot(pts,aes(Dist,PP,colour=Electrode))+geom_point()+scale_colour_manual(values=COLORS)+theme(legend.position='bottom'))
 f3 <- plot_grid(add_axis_labels(plot_grid(p1,p2,p3,ncol=1),'Peak-to-Peak Voltage (mV)'),legend,ncol=1,rel_heights=c(1,.08))
 save_figure(f3,'Figure3_signal_patterns',10,16)
 panels <- map(seq_len(nrow(specs)),function(i){m<-specs$model[i];plot(specs$fit_mcp[[i]])+theme_classic(base_size=16)+labs(title=as.character(specs$Electrode[i]),x=if(m=='Power-law')'Log distance from scar boundary (mm)' else 'Distance from scar boundary (mm)',y='Log peak-to-peak voltage (mV)')})
 f4 <- wrap_plots(panels,ncol=2)+plot_annotation(title=paste('Bayesian piecewise regression — gap',gap,'mm'))
 save_figure(f4,'Figure4_bayesian_piecewise',14,10)
 # Reuse original Figure 5 geometry and palette; lm supplies fixed coefficients.
 fig5_code <- gsub('lme4::fixef\\(fit\\)','coef(fit)',chunk('figure5'))
 fig5_code <- sub('paste(elec, "-", if (loc == "Inferior") "Inferior Wall" else "Roof")','elec',fig5_code,fixed=TRUE)
 fig5_code <- sub('size = 25, hjust = 0.5','size = 16, hjust = 0.5',fig5_code,fixed=TRUE)
 eval(parse(text=fig5_code))
 ggsave(file.path(FIG_DIR,'Figure3_signal_patterns.png'),f3,width=10,height=16,dpi=150,bg='white')
 ggsave(file.path(FIG_DIR,'Figure4_bayesian_piecewise.png'),f4,width=14,height=10,dpi=150,bg='white')
 ggsave(file.path(FIG_DIR,'Figure5_effective_field_of_view.png'),figure5,width=11,height=16,dpi=150,bg='white')
 saveRDS(list(specs=specs,bayes=bayes,ds=ds),file.path(root,paste0('gap_',gap,'mm'),'analysis.rds'))
}
write.csv(bind_rows(summaries),file.path(TAB_DIR,'all_bayesian_summaries.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(root,'sessionInfo.txt'))

writeLines(capture.output(warnings()),file.path(root,"warnings.txt"))
