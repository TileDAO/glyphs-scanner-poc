import "./App.css";

import * as tf from "@tensorflow/tfjs";
import { ChangeEvent, useState } from "react";
import { BigNumber } from "@ethersproject/bignumber";

declare global {
  interface Window {
    tmImage: any;
  }
}

const URL = "https://teachablemachine.withgoogle.com/models/Lfeu6QBi4/";
const modelURL = URL + "model.json";
const metadataURL = URL + "metadata.json";

const canvasIsBlack = (canvas: HTMLCanvasElement) => {
  // getting image data of canvas
  const imgData = canvas
    .getContext("2d")
    ?.getImageData(0, 0, canvas.width, canvas.height);

  // get sum of r+g+b+a of all pixels
  const sum = imgData?.data.reduce(function (a, b) {
    return a + b;
  }, 0);

  if (sum === undefined) return;

  return sum < 250000;
};

function App() {
  const [result, setResult] = useState<string>();
  const [loading, setLoading] = useState<boolean>();

  async function onFileInput(v: ChangeEvent<HTMLInputElement>) {
    if (!v.target.files?.length) return;

    const img = new Image();

    const fr = new FileReader();
    fr.onload = function () {
      if (typeof fr.result !== "string") return;
      img.src = fr.result;
    };
    fr.readAsDataURL(v.target.files[0]);

    img.onload = () => {
      const tensor = tf.browser.fromPixels(img);
      const size = {
        h: tensor.shape[0],
        w: tensor.shape[1],
      };
      const lineWidth = size.w / 34;
      const edgeWidth = lineWidth * 2;
      const blockSize = lineWidth * 10;

      const intTensor = tf.reshape(tensor, [1, size.w, size.h, 3]);

      const floatTensor = intTensor.div<tf.Tensor<tf.Rank.R4>>(255);

      const resizeTfFromRect = (rect: {
        w: number;
        h: number;
        x: number;
        y: number;
      }) =>
        tf.image.cropAndResize(
          floatTensor,
          [
            [
              rect.y / size.h,
              rect.x / size.w,
              (rect.y + rect.h) / size.h,
              (rect.x + rect.w) / size.w,
            ],
          ],
          [0],
          [rect.h, rect.w]
        );

      // borders
      const bwh = {
        w: lineWidth,
        h: lineWidth,
      };
      const borders = [
        resizeTfFromRect({
          ...bwh,
          x: edgeWidth + blockSize,
          y: lineWidth,
        }),
        resizeTfFromRect({
          ...bwh,
          x: edgeWidth + blockSize * 2,
          y: lineWidth,
        }),
        resizeTfFromRect({
          ...bwh,
          x: edgeWidth + blockSize * 3,
          y: edgeWidth + blockSize,
        }),
        resizeTfFromRect({
          ...bwh,
          x: edgeWidth + blockSize * 3,
          y: edgeWidth + blockSize * 2,
        }),
        resizeTfFromRect({
          ...bwh,
          x: edgeWidth + blockSize * 2,
          y: edgeWidth + blockSize * 3,
        }),
        resizeTfFromRect({
          ...bwh,
          x: edgeWidth + blockSize,
          y: edgeWidth + blockSize * 3,
        }),
        resizeTfFromRect({
          ...bwh,
          x: lineWidth,
          y: edgeWidth + blockSize * 2,
        }),
        resizeTfFromRect({
          ...bwh,
          x: lineWidth,
          y: edgeWidth + blockSize,
        }),
      ];

      borders.forEach((b, i) =>
        tf.browser.toPixels(
          tf.reshape<tf.Rank.R3>(b, [b.shape[1], b.shape[2], 3]),
          document.getElementById("b-" + i) as HTMLCanvasElement
        )
      );

      // grids
      for (let index = 0; index < 9; index++) {
        let rects: Record<
          number,
          {
            x: number;
            y: number;
            w: number;
            h: number;
          }
        > = {};

        const wh = {
          w: lineWidth * 5,
          h: lineWidth * 5,
        };

        let _x: number;
        let _y: number;

        switch (index) {
          case 0:
          case 1:
          case 2:
            _x = edgeWidth + blockSize * index;
            _y = edgeWidth;

            rects = {
              0: {
                ...wh,
                x: _x,
                y: _y,
              },
              1: {
                ...wh,
                x: _x + wh.w,
                y: _y,
              },
              2: {
                ...wh,
                x: _x,
                y: _y + wh.h,
              },
              3: {
                ...wh,
                x: _x + wh.w,
                y: _y + wh.h,
              },
            };
            break;
          case 3:
          case 4:
          case 5:
            _x = edgeWidth + blockSize * (index % 3);
            _y = edgeWidth + blockSize;

            rects = {
              0: {
                ...wh,
                x: _x,
                y: _y,
              },
              1: {
                ...wh,
                x: _x + wh.w,
                y: _y,
              },
              2: {
                ...wh,
                x: _x,
                y: _y + wh.h,
              },
              3: {
                ...wh,
                x: _x + wh.w,
                y: _y + wh.h,
              },
            };
            break;
          case 6:
          case 7:
          case 8:
            _x = edgeWidth + blockSize * (index % 3);
            _y = edgeWidth + blockSize * 2;

            rects = {
              0: {
                ...wh,
                x: _x,
                y: _y,
              },
              1: {
                ...wh,
                x: _x + wh.w,
                y: _y,
              },
              2: {
                ...wh,
                x: _x,
                y: _y + wh.h,
              },
              3: {
                ...wh,
                x: _x + wh.w,
                y: _y + wh.h,
              },
            };
            break;
        }

        Object.values(rects).map((rect, i) => {
          let cropped = tf.image.cropAndResize(
            floatTensor,
            [
              [
                rect.y / size.h,
                rect.x / size.w,
                (rect.y + rect.h) / size.h,
                (rect.x + rect.w) / size.w,
              ],
            ],
            [0],
            [rect.h, rect.w]
          );

          if (i % 4 === 1) {
            cropped = tf.reverse4d(cropped, [2]);
          } else if (i % 4 === 2) {
            cropped = tf.reverse4d(cropped, [1]);
          } else if (i % 4 === 3) {
            cropped = tf.image.flipLeftRight(cropped);
            cropped = tf.reverse4d(cropped, [1]);
          }

          tf.browser.toPixels(
            tf.reshape<tf.Rank.R3>(cropped, [
              cropped.shape[1],
              cropped.shape[2],
              3,
            ]),
            document.getElementById(index + "-" + i) as HTMLCanvasElement
          );
        });
      }

      const canvas = document.getElementById("uploaded") as HTMLCanvasElement;
      tf.browser.toPixels(tensor, canvas);

      scan();
    };
  }

  let grids: JSX.Element[] = [];

  for (let i = 0; i < 9; i++) {
    let canvases: JSX.Element[] = [];

    for (let j = 0; j < 4; j++) {
      canvases.push(
        <div
          style={{
            border: "1px solid red",
            maxWidth: 100,
            maxHeight: 100,
          }}
        >
          <canvas id={i + "-" + j}></canvas>
        </div>
      );
    }

    grids.push(
      <div
        style={{
          display: "inline-grid",
          gridTemplate: "1fr 1fr / 1fr 1fr",
          border: "2px solid cyan",
          padding: 5,
          gridGap: 5,
        }}
      >
        {canvases}
      </div>
    );
  }

  let borders: JSX.Element[] = [];

  for (let i = 0; i < 8; i++) {
    borders.push(
      <canvas
        style={{
          border: "1px solid red",
          maxWidth: 30,
          maxHeight: 30,
        }}
        id={"b-" + i}
      ></canvas>
    );
  }

  async function scan() {
    setLoading(true);

    let rotates: string = "";
    let seed: string = "";
    let borderSeeds: string = "";

    const model = await window.tmImage.load(modelURL, metadataURL);

    for (let i = 0; i < 8; i++) {
      const canvas = document.getElementById("b-" + i) as HTMLCanvasElement;
      borderSeeds += canvasIsBlack(canvas) ? "1" : 0;
    }

    for (let i = 0; i < 9; i++) {
      const intFromClassName = (className: string) =>
        parseInt(className.slice(1));

      const hexFromInt = (int: number) =>
        BigNumber.from(int).toHexString().replace("0x0", "").replace("0x", "");

      const readPrediction = (
        predictions: { className: string; probability: number }[]
      ) => {
        const sorted = predictions.sort((a, b) =>
          a.probability > b.probability ? -1 : 1
        );
        const result = sorted[0];
        return result.className;
      };

      const className0 = readPrediction(
        await model.predict(document.getElementById(i + "-0"))
      );
      const className1 = readPrediction(
        await model.predict(document.getElementById(i + "-1"))
      );
      const className2 = readPrediction(
        await model.predict(document.getElementById(i + "-2"))
      );
      const className3 = readPrediction(
        await model.predict(document.getElementById(i + "-3"))
      );

      const int0 = intFromClassName(className0);
      const int1 = intFromClassName(className1);
      const int2 = intFromClassName(className2);
      const int3 = intFromClassName(className3);

      const rotated = className0.startsWith("b");

      let char0: string;
      let char1: string;
      let char2: string;
      let char3: string;

      if (rotated) {
        char0 = hexFromInt(int0 + (int1 >= 16 ? 8 : 0));
        char1 = hexFromInt(int3 + (int2 >= 16 ? 8 : 0));
        char2 = hexFromInt(int1 % 16);
        char3 = hexFromInt(int2 % 16);
      } else {
        char0 = hexFromInt(int1 + (int0 >= 16 ? 8 : 0));
        char1 = hexFromInt(int2 + (int3 >= 16 ? 8 : 0));
        char2 = hexFromInt(int0 % 16);
        char3 = hexFromInt(int3 % 16);
      }

      const newSegment = char0 + char1 + char2 + char3;

      seed += newSegment;

      if (i > 0) rotates += rotated ? "1" : "0";
    }

    const result =
      parseInt(rotates.slice(0, 4), 2).toString(16) +
      parseInt(rotates.slice(4), 2).toString(16) +
      parseInt(borderSeeds.slice(0, 4), 2).toString(16) +
      parseInt(borderSeeds.slice(4), 2).toString(16) +
      seed;

    setLoading(false);
    setResult(result);
  }

  return (
    <div className="App">
      <div id="webcam-container"></div>
      <div id="label-container"></div>

      <div style={{ marginTop: 50, marginBottom: 50 }}>
        <canvas
          style={{ maxWidth: 360, maxHeight: 360 }}
          id="uploaded"
        ></canvas>
      </div>

      <div>
        <label
          htmlFor="img"
          style={{
            border: "1px solid white",
            padding: 8,
            cursor: loading ? "wait" : "pointer",
            opacity: loading ? 0.3 : 1,
          }}
        >
          {loading ? "Scanning..." : "Upload QRT to scan"}
        </label>
        <div style={{ marginTop: 20 }}>
          <a
            href="https://glyph-generator.on.fleek.co"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "white", opacity: 0.6 }}
          >
            QRT generator
          </a>
        </div>
        <input type="file" id="img" onChange={onFileInput} hidden />
        <br />
        <br />
        <br />
        <br />
        {result && (
          <div>
            Scan result:
            <div> 0x{result}</div>
          </div>
        )}
      </div>

      <div
        style={{
          margin: "200px auto 0px",
          opacity: result ? 1 : 0,
        }}
      >
        <div>Parsed:</div>
        <br />
        <br />

        <div
          style={{
            margin: "0 auto",
            display: "flex",
            maxWidth: 400,
            justifyContent: "space-between",
          }}
        >
          {borders}
        </div>

        <div
          style={{
            marginTop: 20,
            paddingBottom: 20,
            display: "inline-grid",
            gridTemplateRows: "1fr 1fr 1fr",
            gridTemplateColumns: "1fr 1fr 1fr",
            gridGap: 20,
          }}
        >
          {grids}
        </div>
      </div>
    </div>
  );
}

export default App;
