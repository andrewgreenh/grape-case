const fs = require('fs')

const logs = fs.readFileSync('./remote-results.log').toString('utf8').split('\n');

const inferenceTimes = logs.filter(line => line.includes('Inference')).map(line => {
  const [, count, seconds] = line.match(/(\d+).*?(\d+(\.\d+)?) seconds/)
  return (+seconds) / (+count);
})

const batchSize = +(process.argv[2] || '32')
const trainingTimesPerImage = logs.filter(line => line.includes('/step')).map(line => {
  const totalSeconds = line.match(/\] - (\d+)s/)[1]
  const batchCount = line.match(/(\d+)\/\1/)[1]

  return (+totalSeconds / +batchCount) / batchSize
})



const meanInferenceTime = mean(inferenceTimes)

console.log('mean inference per image', meanInferenceTime)
console.log('mean training time per image', mean(trainingTimesPerImage))

function mean(ns) {
  return ns.reduce((a, b) => a+b) / ns.length;
}