const {ScriptModule} = require('@pytorch/torchscript');
const fs = require('fs');

const buffer = fs.readFileSync('model.pt');
const model = new ScriptModule(buffer);

const input = new Float32Array([1.0, 2.0]);
const output = model.forward([input]).toTensor();
console.log(output.data);

