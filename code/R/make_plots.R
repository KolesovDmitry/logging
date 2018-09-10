
train_score = read.table('train_score.csv', sep=',')
train_sizes = read.table('train_sizes.csv', sep=',')
val_score = read.table('validation.csv', sep=',')

train_summary = data.frame(size=train_sizes$V1, min=apply(train_score, 1, min), max=apply(train_score, 1, max), mean=apply(train_score, 1, mean))
val_summary = data.frame(size=train_sizes$V1, min=apply(val_score, 1, min), max=apply(val_score, 1, max), mean=apply(val_score, 1, mean))

png('plot.png')
color=3; type = 3
plot(
  val_summary$size, val_summary$mean, type='b',
  ylim=c(0.75, 1),
  col=color,
  xlab="Число пикселей в обучающей выборке",
  ylab="Точность классификатора на тествоом множестве (AUC)"
)
lines(val_summary$size, val_summary$min, lty=type, col=color)
lines(val_summary$size, val_summary$max, lty=type, col=color)
color=4; type = 1
lines(train_summary$size, train_summary$mean, lty=type, col=color)
color=4; type = 2
lines(train_summary$size, train_summary$min, lty=type, col=color)
lines(train_summary$size, train_summary$max, lty=type, col=color)
title(main="Зависимость качества от числа примеров выборки")
legend(
  "bottomright", 
  c('Качество на тестовой выборке', 'Разброс значений на тестовой выборке', NA, 'Качество на обучающей выборке', 'Разброс значений на обучающей выборке'),
  lty=c(1, 3, NA, 1, 2), col=c(3, 3, NA, 4, 4), pch=c(1, NA, NA, NA, NA)
)
dev.off()




