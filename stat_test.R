library("rjson")

fit_result <- fromJSON(file="fitness.json")
df <- as.data.frame(fit_result)

# analyzing CharGen results
chargen <- df[df$Model == "CharGen",]
# print(chargen)
print(kruskal.test(Fitness ~ Temperature, data=chargen))
#print(pairwise.wilcox.test(chargen$Fitness, chargen$Temperature, p.adjust.method="BH"))
#print(chargen[(chargen$Temperature==1)&chargen$SequenceIndex==9,])

tokengen <- df[df$Model == "TokenGen",]
print(kruskal.test(Fitness ~ Temperature, data=tokengen))

print(nrow(tokengen[(tokengen$Executability==1),]))
print(tokengen[order(-tokengen$Fitness),])