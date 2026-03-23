.PHONY: build mdgen html pdf clean all

PANDOC_FLAGS = --standalone --toc --mathjax \
    --metadata title="COMONet: Formal Proofs of Shape Constraint Guarantees" \
    --metadata author="Simon Frost" \
    --highlight-style=tango

# Order matches COMONetProofs.lean imports (dependency order)
MD_FILES = \
    Basics.md Monotonicity.md Convexity.md Concavity.md Composition.md

all: build mdgen html

# Build the Lean 4 project (verifies all proofs)
build:
	lake build

# Generate markdown from Lean source via mdgen
mdgen:
	rm -rf .mdgen_tmp
	mkdir -p .mdgen_tmp
	lake exe mdgen COMONetProofs .mdgen_tmp
	cat $(addprefix .mdgen_tmp/, $(MD_FILES)) > COMONetProofs.md
	rm -rf .mdgen_tmp
	@echo "Generated COMONetProofs.md"

# Generate HTML from markdown
html: COMONetProofs.md
	pandoc COMONetProofs.md -o COMONetProofs.html $(PANDOC_FLAGS)
	@echo "Generated COMONetProofs.html"

# Generate PDF from markdown
pdf: COMONetProofs.md
	pandoc COMONetProofs.md -o COMONetProofs.pdf \
	    --pdf-engine=lualatex \
	    --toc \
	    --metadata title="COMONet: Formal Proofs of Shape Constraint Guarantees" \
	    --metadata author="Simon Frost" \
	    --highlight-style=tango \
	    -H header.tex
	@echo "Generated COMONetProofs.pdf"

clean:
	rm -f COMONetProofs.html COMONetProofs.pdf COMONetProofs.md
	rm -rf .mdgen_tmp
