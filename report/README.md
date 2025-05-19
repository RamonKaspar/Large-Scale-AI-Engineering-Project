## Rendering

The report was renderer using the [Eisvogel template](https://github.com/Wandmalfarbe/pandoc-latex-template).

```bash
pandoc report_pretokenization.md -o report_pretokenization.pdf --template ./eisvogel.latex

pandoc report_ddp.md -o report_ddp.pdf --template ./eisvogel.latex
```