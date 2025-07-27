require('dotenv').config();
const express = require("express");
const fs = require("fs");
const path = require("path");
const { execFile } = require("child_process");
const csv = require('csv-parser');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT;
app.use(express.json());
app.use(cors());

const PYTHON_PATH = "../venv/bin/python3.10";

function cleanFiles() {
  const filesToDelete = ["explanations.json", "explanation_report.json", "results.json"];
  for (let file of filesToDelete) {
    const fPath = path.join(__dirname, file);
    if (fs.existsSync(fPath)) fs.unlinkSync(fPath);
  }
}

function runPythonScripts(req, res) {
  cleanFiles();

  const script1 = path.join(__dirname, "P2_explain_careerCA.py");
  const script2 = path.join(__dirname, "P2_generate_reportCB.py");

  execFile(PYTHON_PATH, [script1], (err1, stdout1, stderr1) => {
    if (err1) {
      return res.status(500).json({ error: "Error in Step 1", details: stderr1 || err1.message });
    }

    execFile(PYTHON_PATH, [script2], (err2, stdout2, stderr2) => {
      if (err2) {
        return res.status(500).json({ error: "Error in Step 2", details: stderr2 || err2.message });
      }

      const resultPath = path.join(__dirname, "explanation_report.json");
      if (fs.existsSync(resultPath)) {
        const data = fs.readFileSync(resultPath, "utf-8");
        return res.status(200).json(JSON.parse(data));
      } else {
        return res.status(500).json({ error: "explanations.json not generated." });
      }
    });
  });
}

app.get("/process", runPythonScripts);

app.get("/start", (req, res) => {
  const filesToDelete = ["input.json"];
  for (let file of filesToDelete) {
    const fPath = path.join(__dirname, file);
    if (fs.existsSync(fPath)) fs.unlinkSync(fPath);
  }
  return res.status(200).json({"message":"files cleared"});
})

app.post("/input", (req, res) => {
  const data = req.body;
  if (!data || Object.keys(data).length === 0) {
    return res.status(400).json({ error: "No input data provided." });
  }

  const inputPath = path.join(__dirname, "input.json");

  try {
    fs.writeFileSync(inputPath, JSON.stringify(data, null, 2));
    return res.status(200).json({ message: "✅ input.json saved successfully." });
  } catch (err) {
    return res.status(500).json({ error: "Failed to write input.json", details: err.message });
  }
});

app.get('/list', (req, res) => {
  const csvFilePath = 'final_dataset_with_skill_clusters.csv';
  const skipColumns = ['label', 'skill_cluster']; 
  let responseSent = false;

  const stream = fs.createReadStream(csvFilePath)
    .pipe(csv())
    .on('headers', (csvHeaders) => {
      const filteredHeaders = csvHeaders.filter(header => !skipColumns.includes(header));
      if (!responseSent) {
        res.json({ columns: filteredHeaders });
        responseSent = true;
        stream.destroy(); 
      }
    })
    .on('error', (err) => {
      if (!responseSent) {
        res.status(500).json({ error: 'Failed to read CSV file', details: err.message });
        responseSent = true;
      }
    });
});


app.listen(PORT, () => {
  console.log(`🚀 Backend running on http://localhost:${PORT}`);
});
