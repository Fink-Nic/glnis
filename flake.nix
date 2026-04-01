{
  description = "glNIS - Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Define the libraries your Python binaries are looking for
        libs = with pkgs; [
          stdenv.cc.cc.lib  # This provides libstdc++.so.6
          zlib
          gmp
          mpfr
          libmpc
          python313
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            gcc
            uv
            maturin
            # --- Rust Toolchain ---
            rustc
            cargo
            rust-analyzer
            clippy
            rustfmt
          ];

          buildInputs = libs;

          shellHook = ''
            # Dynamically build the path from the 'libs' list
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath libs}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            
            echo "Rust $(rustc --version) and Cargo loaded."
            echo "Libraries loaded. libstdc++.so.6 is now visible to your .venv."
          '';
        };
      }
    );
}