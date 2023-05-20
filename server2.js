const dgram = require('dgram');
const fs = require('fs');
const server = dgram.createSocket('udp4');
const Writable = require('stream').Writable;
const speaker = new Writable();

// Factor de escala para ajustar el volumen. Un valor de 1.0 no altera el volumen.
// Valores mayores aumentan el volumen, valores menores lo disminuyen.
// Ten cuidado al elegir este valor, ya que valores demasiado altos pueden causar distorsión.
const volumeScale = 1.5;

speaker._write = function (chunk, enc, next) {
  let buf = Buffer.alloc(chunk.length);
  for (let i = 0; i < chunk.length; i += 2) {
    let value = chunk.readInt16LE(i);
    value *= volumeScale; // Ajusta el volumen
    // Asegúrate de que el valor no exceda los límites de un entero de 16 bits
    value = Math.min(Math.max(value, -32768), 32767);
    buf.writeInt16LE(value, i);
  }
  fs.appendFileSync('audio.raw', buf);
  next();
};

server.on('error', (err) => {
  console.log(`server error:\n${err.stack}`);
  server.close();
});

server.on('message', (msg, rinfo) => {
  console.log(`server got: ${msg} from ${rinfo.address}:${rinfo.port}`);
  speaker.write(msg);
});

server.on('listening', () => {
  const address = server.address();
  console.log(`server listening ${address.address}:${address.port}`);
});

server.bind(12345); // Reemplaza con el puerto que prefieras
