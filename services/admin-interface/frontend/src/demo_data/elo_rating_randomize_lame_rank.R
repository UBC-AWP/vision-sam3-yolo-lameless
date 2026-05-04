library(EloRating)
library(RColorBrewer)
library(lubridate)
library(tidyr)
source("../../08-Lameness_rank_eloSteepness/code/eloSteepness_helpers.R")

# load in the data
expert_dir <- "../../05-Amazon_MTurk_expert_response_30cow_pairwise/results/all_experts/"
winner_loser <- read.csv(paste0(expert_dir, "winner_loser_merged_DW_NV_SB_KI.csv"), header = TRUE, sep = ",")

# set output directory
output_dir <- "../results/"

# replicate row is degree > 1
winner_loser_degree_replct <- replicate_row_df(winner_loser)

# load experts' eloSteepness results
expert_elo_dir <-"../../08-Lameness_rank_eloSteepness/results/"
expert_eloSteep <- read.csv(paste0(expert_elo_dir, "compare_summary.csv"), header = TRUE, sep = ",")
expert_eloSteep$X <- NULL
expert_eloSteep2 <- expert_eloSteep[, c("Cow", "NV_DW_SB_KI_experts_mean")]


#####################################################################
############ random Elo-rating with individual raw response #########
#####################################################################
# randomize the sequence 100 times
# Initialize a matrix to store the Elo ratings for each iteration
set.seed(123)

# Loop over 100 iterations
calculate_elo_ratings <- function(interaction_df, num_iterations=100, consider_weight=FALSE, my_weight=NULL) {
  for (i in 1:num_iterations) {
    # Randomize the sequence of rows in interaction_df
    interaction_df_random <- interaction_df[sample(nrow(interaction_df)), ]
    
    # Process the data as before
    interaction_df_random$Date <-  as.character("2025-01-02")
    interaction_df_random$Date[1:round(nrow(interaction_df_random)/2)] <- as.character("2025-01-01")
    interaction_df_processed_elo <- interaction_df_random
    interaction_df_processed_elo$tie = FALSE
    interaction_df_processed_elo$tie[which(interaction_df_processed_elo$degree == 0)] <- TRUE
    interaction_df_processed_elo <- interaction_df_processed_elo[, c( "Date", "winner", "loser","tie", "degree")]
    
    # Calculate the Elo ratings
    if (consider_weight) {
      elo_package_result = elo.seq(winner = as.character(interaction_df_processed_elo$winner),
                                   loser = as.character(interaction_df_processed_elo$loser),
                                   Date = interaction_df_processed_elo$Date,
                                   draw = interaction_df_processed_elo$tie,
                                   k = my_weight,
                                   intensity = as.character(interaction_df_processed_elo$degree),
                                   runcheck = F,
                                   progressbar = T)
    } else {
      elo_package_result = elo.seq(winner = as.character(interaction_df_processed_elo$winner),
                                   loser = as.character(interaction_df_processed_elo$loser),
                                   Date = interaction_df_processed_elo$Date,
                                   draw = interaction_df_processed_elo$tie,
                                   k = 20,
                                   runcheck = F,
                                   progressbar = T)
    }
    
    elo_mat <- as.data.frame(elo_package_result$mat)
    elo_mat <- cbind(as.Date(elo_package_result$truedates), elo_mat)
    colnames(elo_mat)[1] <- "Date"
    
    ##Transform data to long format for plotting (Date, CowID, ELO)
    keycol <- "Cow"
    valuecol <- "Elo"
    gathercols <- colnames(elo_mat)[2:length(colnames(elo_mat))]
    elo_mat_long <- gather(elo_mat, keycol, valuecol, gathercols)
    colnames(elo_mat_long) <- c("Date", "Cow", paste("Elo_", i, sep = ""))
    elo_mat_long2 <- elo_mat_long[which(elo_mat_long$Date == "2025-01-02"),]
    elo_mat_long3 <- elo_mat_long2[, c("Cow", paste("Elo_", i, sep = ""))]
    elo_mat_long3 <- elo_mat_long3[order(elo_mat_long3$Cow),]
    rownames(elo_mat_long3) <- NULL
    
    if (i == 1) {
      elo_ratings = elo_mat_long3
    } else {
      elo_ratings <- merge(elo_ratings, elo_mat_long3, by = "Cow", all.x = TRUE)
    }
  }
  
  return(elo_ratings)
}

elo_ratings <- calculate_elo_ratings(winner_loser_degree_replct, 1)
elo_ratings_Elo_only <- elo_ratings[, 2:ncol(elo_ratings)]

# Calculate the average Elo rating for each cow
average_elo_ratings <- rowMeans(elo_ratings_Elo_only, na.rm = TRUE)

# Create a data frame of the average Elo ratings
average_elo_df <- data.frame(Cow = elo_ratings$Cow, Elo_random = elo_ratings_Elo_only)
master <- merge(average_elo_df, expert_eloSteep2)

output_file <- paste0(output_dir, "elo_rating_VS_eloSteepness.csv")
write.csv(master, file = output_file, row.names = FALSE)
