--[[
ieee.lua — pandoc filter used by docs/md_to_pdf.py --style ieee.

Turns the pandoc AST of an ENSy document into IEEEtran constructs:
  * figures            -> figure* floats at text width (top of page, as in LaTeX)
  * ::: {.booktabs}    -> table / table* floats with booktabs rules; the caption
                          ("Table: ...") goes above, notes after the table go
                          inside the float, long columns become tabularx X columns
  * display math       -> numbered equation environments
  * code blocks        -> full text width in place (cuted strip), small type
  * ::: {#ieee-refs}   -> thebibliography, one \bibitem{rN} per "[N] ..." entry
]]

local function latex(blocks)
  local out = pandoc.write(pandoc.Pandoc(blocks), "latex")
  return (out:gsub("%s+$", ""))
end

local function inline_latex(blocks)
  -- a caption or cell rendered without paragraph breaks
  return (latex(blocks):gsub("\n\n+", " "))
end

-- ── figures ────────────────────────────────────────────────────────────
function Figure(fig)
  local src
  pandoc.walk_block(fig, { Image = function(img) src = src or img.src end })
  if not src then return nil end
  local caption = inline_latex(fig.caption.long)
  return pandoc.RawBlock("latex", table.concat({
    "\\begin{figure*}[tp]",
    "\\centering",
    "\\includegraphics[width=\\textwidth,height=0.42\\textheight,keepaspectratio]{" .. src .. "}",
    "\\caption{" .. caption .. "}",
    "\\end{figure*}",
  }, "\n"))
end

-- ── tables ─────────────────────────────────────────────────────────────
local ALIGN = { AlignLeft = "l", AlignRight = "r", AlignCenter = "c", AlignDefault = "l" }

-- monospace identifiers in cells may break at / . _ (url's \path)
local function breakable_code(blocks)
  return pandoc.walk_block(pandoc.Div(blocks), {
    Code = function(c)
      for _, d in ipairs({ "|", "!", "+", "@", "=" }) do
        if not c.text:find(d, 1, true) then
          return pandoc.RawInline("latex", "\\path" .. d .. c.text .. d)
        end
      end
    end,
  }).content
end

local function render_table(tbl, wide)
  local ncols = #tbl.colspecs
  local rows, longest = {}, {}
  for i = 1, ncols do longest[i] = 0 end

  local function add_rows(list, is_head)
    for _, row in ipairs(list) do
      local cells = {}
      for i, cell in ipairs(row.cells) do
        local text = inline_latex(breakable_code(cell.contents))
        cells[#cells + 1] = text
        local plain = pandoc.utils.stringify(cell.contents)
        if not is_head and #plain > longest[i] then longest[i] = #plain end
      end
      rows[#rows + 1] = { cells = cells, head = is_head }
    end
  end
  add_rows(tbl.head.rows, true)
  for _, body in ipairs(tbl.bodies) do
    add_rows(body.head, false)
    add_rows(body.body, false)
  end

  -- the longest column wraps (tabularx X) when its content is long
  local xcol, xlen = nil, 0
  for i = 1, ncols do
    if longest[i] > xlen then xcol, xlen = i, longest[i] end
  end
  local use_x = xlen > (wide and 60 or 28)
  local spec = {}
  for i, cs in ipairs(tbl.colspecs) do
    -- wide tables: only the longest column wraps; narrow tables: every long column
    if use_x and (i == xcol or (not wide and longest[i] > 14)) then
      spec[#spec + 1] = ">{\\raggedright\\arraybackslash}X"
    else
      spec[#spec + 1] = ALIGN[cs[1]] or "l"
    end
  end

  local width = wide and "\\textwidth" or "\\columnwidth"
  local env = use_x and "tabularx" or "tabular*"
  local open = use_x and ("\\begin{tabularx}{" .. width .. "}{" .. table.concat(spec) .. "}")
                     or ("\\begin{tabular*}{" .. width .. "}{@{\\extracolsep{\\fill}}" .. table.concat(spec) .. "@{}}")
  local lines = { open, "\\toprule" }
  local seen_head = false
  for _, row in ipairs(rows) do
    if not row.head and not seen_head then
      lines[#lines + 1] = "\\midrule"
      seen_head = true
    end
    lines[#lines + 1] = table.concat(row.cells, " & ") .. " \\\\"
  end
  lines[#lines + 1] = "\\bottomrule"
  lines[#lines + 1] = "\\end{" .. env .. "}"
  return table.concat(lines, "\n")
end

function Div(div)
  if div.classes:includes("booktabs") then
    local wide = div.classes:includes("wide")
    local tbl, notes = nil, {}
    for _, block in ipairs(div.content) do
      if block.t == "Table" and not tbl then
        tbl = block
      elseif tbl then
        notes[#notes + 1] = block
      end
    end
    if not tbl then return nil end
    local env = wide and "table*" or "table"
    local parts = { "\\begin{" .. env .. "}[tp]", "\\centering" }
    local caption = inline_latex(tbl.caption.long)
    if caption ~= "" then parts[#parts + 1] = "\\caption{" .. caption .. "}" end
    parts[#parts + 1] = "\\footnotesize"
    parts[#parts + 1] = render_table(tbl, wide)
    if #notes > 0 then
      parts[#parts + 1] = "\\par\\vspace{3pt}\\parbox{" .. (wide and "\\textwidth" or "\\columnwidth")
        .. "}{\\scriptsize " .. inline_latex(notes) .. "}"
    end
    parts[#parts + 1] = "\\end{" .. env .. "}"
    return pandoc.RawBlock("latex", table.concat(parts, "\n"))
  end

  if div.identifier == "ieee-refs" then
    local items = { "\\begin{thebibliography}{" .. "99" .. "}" }
    for _, block in ipairs(div.content) do
      if block.t == "Para" then
        local first = block.content[1]
        local n = first and first.t == "Str" and first.text:match("^%[(%d+)%]$")
        if n then
          local rest = pandoc.List(block.content)
          rest:remove(1)
          if rest[1] and rest[1].t == "Space" then rest:remove(1) end
          items[#items + 1] = "\\bibitem{r" .. n .. "} " .. inline_latex({ pandoc.Plain(rest) })
        end
      end
    end
    items[#items + 1] = "\\end{thebibliography}"
    return pandoc.RawBlock("latex", table.concat(items, "\n"))
  end
end

-- ── equations and code ─────────────────────────────────────────────────
function Math(m)
  if m.mathtype == "DisplayMath" then
    return pandoc.RawInline("latex", "\\begin{equation}" .. m.text .. "\\end{equation}")
  end
end

function CodeBlock(cb)
  -- full text width at this point of the text (cuted's strip), not a float
  return pandoc.RawBlock("latex", table.concat({
    "\\begin{strip}",
    "\\begin{minipage}{\\textwidth}",  -- never split across pages
    "\\footnotesize",
    "\\begin{verbatim}",
    cb.text,
    "\\end{verbatim}",
    "\\end{minipage}",
    "\\end{strip}",
  }, "\n"))
end
