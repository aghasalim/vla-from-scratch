# The same tables are printed in two documents. They have to say the same thing.
#
# scripts/check_numbers.py reads README.md and notes/METHODS.md concatenated,
# so a figure that survives in either one passes. The two documents carry
# duplicate copies of the latency table and the step sweep table, which means a
# number could drift in the README, stay right in the notes, and the drift check
# would still be green. Nothing compared the copies.
#
# This pulls every markdown table out of both documents, matches them by header
# row, and requires the bodies to be identical cell for cell. Emphasis markers
# are stripped first, so bolding a number does not read as the number changing.
#
#   ruby verify/tables.rb

root = ARGV[0] || "."
DOCS = ["README.md", "notes/METHODS.md"].freeze

# Ruby 2.6 opens files as US-ASCII, and these documents are not.
def slurp(path)
  File.read(path, encoding: "UTF-8")
end

def cells(row)
  row.strip.sub(/\A\|/, "").sub(/\|\z/, "")
     .split("|").map { |c| c.strip.gsub(/\*+/, "").gsub(/`/, "") }
end

def tables(text)
  out = {}
  lines = text.lines
  lines.each_with_index do |line, i|
    next unless line.strip.start_with?("|")
    nxt = lines[i + 1]
    next unless nxt && nxt.strip =~ /\A\|[\s:|-]+\|\z/

    header = cells(line)
    body = []
    j = i + 2
    while lines[j] && lines[j].strip.start_with?("|")
      body << cells(lines[j])
      j += 1
    end
    out[header] = body
  end
  out
end

found = DOCS.map { |d| [d, tables(File.join(root, d).then { |p| slurp(p) })] }.to_h
shared = found[DOCS[0]].keys & found[DOCS[1]].keys
failures = []

if shared.empty?
  failures << "the two documents no longer share a single table, so nothing was compared"
end

shared.each do |header|
  a = found[DOCS[0]][header]
  b = found[DOCS[1]][header]
  label = header.join(" | ")
  if a.length != b.length
    failures << "table [#{label}]: README has #{a.length} rows, notes/METHODS.md has #{b.length}"
    next
  end
  a.zip(b).each_with_index do |(ra, rb), i|
    next if ra == rb

    failures << "table [#{label}] row #{i + 1}: README #{ra.inspect} against " \
                "notes/METHODS.md #{rb.inspect}"
  end
end

rows = shared.sum { |h| found[DOCS[0]][h].length }
cells_checked = shared.sum { |h| found[DOCS[0]][h].sum(&:length) }
puts "tables.rb: #{shared.length} tables shared by both documents, " \
     "#{rows} rows, #{cells_checked} cells compared, #{failures.length} failures"
failures.each { |f| puts "  - #{f}" }
exit(failures.empty? ? 0 : 1)
