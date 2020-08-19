const tf = require('@tensorflow/tfjs-node');
// const tf = require("@tensorflow/tfjs");
// const toxicity = require("@tensorflow-models/toxicity");
// const threshold = 0.8;
// toxicity.load(threshold).then(model => {
//   // const sentences = [ 'good one' ];
//   // const sentences = [ 'you suck' ];

//   model.classify("hey dumb guy").then(predictions => {
//     // console.log(JSON.stringify(predictions, null, "  "));
//     predictions.forEach(predict => {
//       if ((Object.values(predict))[ 1 ][ 0 ].match) {
//         console.log("toxic comments arent allowed sorry");
//       }

//     });

//   });
// });

//detect nsfw
const axios = require('axios'); //you can use any http client
const nsfw = require('nsfwjs');
async function fn () {
  const pic = await axios.get(`https://www.filmibeat.com/ph-big/2016/01/south-indian-actress-hot-cleavage_1453378737150.jpg`, {
    responseType: 'arraybuffer',
  });
  const model = await nsfw.load(); // To load a local model, nsfw.load('file://./path/to/model/')
  // Image must be in tf.tensor3d format
  // you can convert image to tf.tensor3d with tf.node.decodeImage(Uint8Array,channels)
  const image = await tf.node.decodeImage(pic.data, 3);
  const predictions = await model.classify(image);
  image.dispose(); // Tensor memory must be managed explicitly (it is not sufficient to let a tf.Tensor go out of scope for its memory to be released).
  console.log(predictions);
}
fn();
