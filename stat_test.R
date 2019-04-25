library("rjson")

fit_result <- fromJSON(file="fitness.json")
df <- as.data.frame(fit_result)

# analyzing CharGen results
chargen <- df[df$Model == "CharGen",]
# print(chargen)
print(kruskal.test(Fitness ~ Temperature, data=chargen))
print(pairwise.wilcox.test(chargen$Fitness, chargen$Temperature, p.adjust.method="BH"))
