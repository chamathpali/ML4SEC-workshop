library(decryptr) # https://rdrr.io/github/decryptr/decryptr/
library(keras)

url <- "http://www.tjrs.jus.br/site_php/consulta/human_check/humancheck_showcode.php"

train <- download_captcha(url, n = 10, path = "./train")

test <- download_captcha(url, n = 10, path = "./test")

new_files <- classify(train, path = "./train")

captchas <- read_captcha(new_files, ans_in_path = TRUE)

mymodel <- train_model(captchas, verbose = FALSE)

decrypt(test, model = mymodel)
