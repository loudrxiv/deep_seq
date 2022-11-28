# PREAMBLE=====
#- Authors
##- Dennis Kostka 
##- Mark Ebeid

# LIBRARIES==== 
library(reticulate)
library(tensorflow)
library(tfdatasets)
library(BSgenome.Hsapiens.UCSC.hg38)
library(ggplot2)
library(keras)
library(tfaddons)

#- Sanity
if(! is_keras_available(version = NULL)) 
  stop("No Keras.")

# CONSTANTS====
HOCOMOCO_DAT    <- "~/data/seq_vae/hocomoco/human/hoc_pwms_hg_16.rdat"

#- Plotting
DIR_PLOTS       <- "~/plots/seq_vae/vae_model/"
DIR_TENSORBOARD <- paste0(DIR_PLOTS, "tensorboards/")

# Functions ====
filterSafetyCheck = function(the_filters) {
  if(any(lapply(the_filters, is.matrix) |> unlist() == FALSE))            
    stop("filters invalid\n")
  if(any(lapply(the_filters, ncol) |> unlist() != ncol(the_filters[[1]]))) 
    stop("filters invalid\n")
  if(any(lapply(the_filters, nrow) |> unlist() != nrow(the_filters[[1]]))) 
    stop("filters invalid\n")
}
reconstruction_loss = function(y_true, y_pred){
   return(
       tf$math$reduce_mean(
           tf$keras$metrics$binary_crossentropy(y_true,
                                                y_pred,
                                                label_smoothing = 1E-5)))
}
kl_divergence = function(z_mean, z_logVar) {
  return( -0.5 * tf$reduce_mean(z_logVar - tf$square(z_mean) - 
                                  tf$exp(z_logVar) + 1) )
}
make_ints = function(seqs){
  tmp      <- as.matrix(seqs)
  seqs_int <- apply(tmp, 2,function(x) as.factor(x) |> as.integer() - 1L)
  return(seqs_int)
}
make_1h = function(seqs){
  seq_ints <- make_ints(seqs)
  seqs_1h  <- tf$one_hot(seqs_int,4L) |> as.array()
  return(seqs_1h)
}

# Load Data ====
#- Generate Tensor (PWSM)
my_filters <- readRDS(HOCOMOCO_DAT) |> lapply(as.matrix)
filterSafetyCheck(my_filters)
filter_tensor <- my_filters |> 
    unlist() |> 
    array(dim = c( nrow(my_filters[[1]]), 4, length(my_filters)))

#- Tile Sequences
mt = GenomicRanges::tileGenome(BSgenome.Hsapiens.UCSC.hg38::Hsapiens |> 
                                 seqinfo() |> keepStandardChromosomes(), 
                               tilewidth=16*1024, 
                               cut.last.tile.in.chrom = TRUE)
#- Format Sequences
sqs        = Biostrings::getSeq(Hsapiens,mt) 
names(sqs) = mt |> as.character()
nNs        = Biostrings::vcountPattern('N',sqs)
sqs        = sqs[nNs == 0]
sqs        = sqs[width(sqs) == 16*1024] 

# MOCEL CREATION====
ipt <- layer_input(shape = list(NULL,4), name = "ipt")
el1 = layer_conv_1d(filters = dim(filter_tensor)[3], 
                    kernel_size = dim(filter_tensor)[1],
                    use_bias = FALSE,
                    name = "el1", 
                    #weights( list(filter_tensor, rep(0, dim(filter_tensor)[3])|> as.array()) ),
                    trainable = FALSE, 
                    padding = "same",
                    input_shape = list(NULL,4))
embed_seq_1 = keras_model_sequential(name = "embed_seq_1")
embed_seq_1$add(el1)

#- Add PWSM weights to model
get_layer(embed_seq_1, name="el1")$set_weights(weights = list(filter_tensor))

#- Sanity
#get_layer(embed_seq_1, name="el1")$get_weights()

ipt_tf = embed_seq_1(ipt)

#- Encoder
embed_seq_2 = keras_model_sequential( name = "embed_seq_2")
embed_seq_2$add(layer_max_pooling_1d(pool_size = 4L) ) #- 4bp resolution, 34 bp locality
embed_seq_2$add(layer_batch_normalization())
embed_seq_2$add(layer_conv_1d(filters = 256, kernel_size = 5, padding = "same", activation = 'relu'))
embed_seq_2$add(layer_max_pooling_1d(pool_size = 2L))  #- 8 bp resoution, 40 bp locality
embed_seq_2$add(layer_batch_normalization())
embed_seq_2$add(layer_conv_1d(filters = 64, kernel_size = 5, padding = "same", activation = 'linear') )
embed_seq_2$add(layer_max_pooling_1d(pool_size = 2L))  #- 16 bp resoution, ?? bp locality
embed_seq_2$add(layer_batch_normalization())

#- Latent Embedding
z = ipt_tf |> embed_seq_2()
z_mean   = z |> layer_conv_1d(filters = 32, 
                                kernel_size = 1L, 
                                padding = "same",
                                use_bias = TRUE)
z_logVar = z |> layer_conv_1d(filters = 32, 
                                kernel_size = 1L, 
                                padding = "same",
                                use_bias = TRUE)
mod_embed = keras_model(ipt,z_mean, name = "embed_seq")

b_dim    = tf$shape(z_mean)[1] # batch dimension
x_dim    = tf$shape(z_mean)[2] # sequence length
y_dim    = tf$shape(z_mean)[3] # latent dimension (z_dim above)
eps      = tf$random$normal(shape = c(b_dim, x_dim, y_dim), seed = 173649)
z_sample =  z_mean + tf$math$exp(0.5 * z_logVar) * eps

#- Decoder 
mod_decode = keras_model_sequential(name = "decode_seq")
mod_decode$add(layer_dropout(rate=0.01))
mod_decode$add(layer_conv_1d_transpose(filters = 64, stride = 2, kernel_size = 8, padding='same', activation = 'relu'))
mod_decode$add(layer_batch_normalization())
mod_decode$add(layer_dropout(rate=0.01))
mod_decode$add(layer_conv_1d_transpose(filters = 64, stride = 2, kernel_size = 8, padding='same', activation = 'relu'))
mod_decode$add(layer_batch_normalization())
mod_decode$add(layer_dropout(rate=0.01))
mod_decode$add(layer_conv_1d_transpose(filters = 64,  stride = 2, kernel_size = 8, padding='same', activation = 'relu'))
mod_decode$add(layer_batch_normalization())
mod_decode$add(layer_dropout(rate=0.01))
mod_decode$add(layer_conv_1d_transpose(filters = 64,  stride = 2, kernel_size = 8, padding='same', activation = 'relu'))
mod_decode$add(layer_batch_normalization())
mod_decode$add(layer_dropout(rate=0.01))
mod_decode$add(layer_conv_1d_transpose(filters = 64, stride = 1, kernel_size = 4,  padding='same',activation = 'relu'))
mod_decode$add(layer_batch_normalization())
mod_decode$add(layer_conv_1d_transpose(filters = 64, stride = 1, kernel_size = 2,  padding='same',activation = 'relu'))
mod_decode$add(layer_batch_normalization())
mod_decode$add(layer_conv_1d_transpose(filters = 4,  kernel_size = 1,   padding='same', activation = 'sigmoid'))

rec = z_sample |> mod_decode()
mod_vae = keras_model(ipt, rec, name="x_vae")

#- KL Divergence Loss
mod_vae$add_loss( kl_divergence(z_mean,z_logVar) * 0.001 )

# MODEL COMPILATION & FITTING====
mod_vae = mod_vae |> compile(loss="binary_crossentropy", 
                       optimizer = keras$optimizers$Adamax(),
                       metrics    = list(tf$keras$metrics$BinaryAccuracy(name="acc"),
                                         tf$keras$metrics$AUC(name="auroc"),
                                         tf$keras$metrics$AUC(name="auprc", curve="PR")))

early_stopping = tf$keras$callbacks$EarlyStopping( monitor  = 'val_auprc', 
                                                    patience =  7,
                                                    restore_best_weights = TRUE, 
                                                    mode = 'max')
#- Shuffle data for better learning
set.seed(24213)
tst_size = length(sqs) %/% 5
val_size = (length(sqs) * 0.8) %/% 5
shuff    = sample(seq_len(sqs |> length() ))

#- Split into train & test sets
ssqs     = sqs[shuff]
ds_test  = ssqs[1:tst_size]        |> make_ints()
ds_train = sqs[-(1:tst_size)]
ds_val   = ds_train[1:val_size]    |> make_ints()
ds_train = ds_train[-(1:val_size)] |> make_ints()

#- Utilize tfdatasets for efficieny
mds_train  =   tensor_slices_dataset(ds_train)       |>
  dataset_map(function(x) tf$one_hot(x,4L), 
              num_parallel_calls = tf$data$AUTOTUNE) |>
  dataset_map(function(x) list(x,x), 
              num_parallel_calls = tf$data$AUTOTUNE) |>
  dataset_shuffle( buffer_size = 10000L, 
                   reshuffle_each_iteration=TRUE)    |> 
  dataset_batch(64, drop_remainder = TRUE)           |>
  dataset_repeat()                                   |>
  dataset_prefetch_to_device(device = '/gpu:0', 
                             buffer_size = tf$data$AUTOTUNE)

mds_val  =     tensor_slices_dataset(ds_val)         |>
  dataset_map(function(x) tf$one_hot(x,4L), 
              num_parallel_calls = tf$data$AUTOTUNE) |>
  dataset_map(function(x) list(x,x), 
              num_parallel_calls = tf$data$AUTOTUNE) |>
  dataset_shuffle( buffer_size = 10000L, 
                   reshuffle_each_iteration=TRUE)    |> 
  dataset_batch(64, drop_remainder = TRUE)           |>
  dataset_repeat()                                   |>
  dataset_prefetch_to_device(device = '/gpu:0', 
                             buffer_size = tf$data$AUTOTUNE)

#- Create Tensorboard
tb <- paste0( DIR_TENSORBOARD, Sys.time() )
tensorboard(tb)

history <- mod_vae |> fit(
  mds_train,
  epochs=200, 
  steps_per_epoch = 50, 
  class_weights = c("0"=1/3, "1"=1), 
  validation_data=mds_val,
  validation_steps=10,
  callbacks = list(early_stopping,callback_tensorboard(tb))
  )

# Save Models ====
save_model_tf(mod_vae,    filepath = "~/seq_vae/src/models/vae_model/vae")
save_model_tf(mod_embed,  filepath = "~/seq_vae/src/models/vae_model/encoder")
save_model_tf(mod_decode, filepath = "~/seq_vae/src/models/vae_model/decoder")