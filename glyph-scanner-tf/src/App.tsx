import "./App.css";

import * as tf from "@tensorflow/tfjs";
import { ChangeEvent } from "react";

function App() {
  function onFileInput(v: ChangeEvent<HTMLInputElement>) {
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
          const cropped = tf.image.cropAndResize(
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

          const canvas = document.getElementById(
            index + "-" + i
          ) as HTMLCanvasElement;

          tf.browser.toPixels(
            tf.reshape<tf.Rank.R3>(cropped, [
              cropped.shape[1],
              cropped.shape[2],
              3,
            ]),
            canvas
          );
        });
      }

      const canvas = document.getElementById("uploaded") as HTMLCanvasElement;
      tf.browser.toPixels(tensor, canvas);
    };
  }

  let grids: JSX.Element[] = [];

  for (let i = 0; i < 9; i++) {
    let canvases: JSX.Element[] = [];

    for (let j = 0; j < 4; j++) {
      canvases.push(
        <canvas style={{ border: "1px solid red" }} id={i + "-" + j}></canvas>
      );
    }

    grids.push(
      <div
        style={{
          display: "inline-grid",
          gridTemplate: "1fr 1fr / 1fr 1fr",
          gridGap: 5,
          border: "2px solid cyan",
        }}
      >
        {canvases}
      </div>
    );
  }

  return (
    <div className="App">
      <div>
        <canvas id="uploaded"></canvas>
      </div>

      <div>
        <input type="file" id="img" onChange={onFileInput} />
      </div>

      <div
        style={{
          paddingTop: 100,
          paddingBottom: 100,
          display: "inline-grid",
          gridTemplateRows: "1fr 1fr 1fr",
          gridTemplateColumns: "1fr 1fr 1fr",
          gridGap: 20,
        }}
      >
        {grids}
      </div>
    </div>
  );
}

export default App;
