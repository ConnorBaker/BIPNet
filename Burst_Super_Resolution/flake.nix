{
  description = "A flake for an ML project";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
  # TODO: arm64 support?
    flake-utils.lib.eachSystem ["x86_64-linux"] (system:
      with import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
        overlays = [
          (final: prev: {
            # AWS EC2 instances can't run anything newer than 5.17 at the moment.
            # You must set your /etc/nixos/configuration.nix to use this linux kernel and nvidia driver.
            linuxPackages_5_15 = prev.linuxPackages_5_15.extend (linuxFinal: linuxPrev: let
              # TODO: Which nixpkgs is this using? The one from inputs?
              generic = args: linuxFinal.callPackage (import <nixpkgs/pkgs/os-specific/linux/nvidia-x11/generic.nix> args) {};
            in {
              nvidiaPackages.beta = generic {
                version = "515.43.04";
                sha256_64bit = "1irx9w69f39sglr69jn01j9fq8lir2kbapdvy8b26jqf6m6mm1ry";
                settingsSha256 = null; # We run in a headless environment
                persistencedSha256 = "0snxvgxkg83kllhnq4vd8hjjzy0xgzpyh2wdy33r4mi79k5w2hl5";
              };
            });

            opencv4 = prev.opencv4.overrideAttrs (oldAttrs: {
              enableFfmpeg = true;
              enableCuda = true;
              enableUnfree = true;

              propagatedBuildInputs = oldAttrs.propagatedBuildInputs ++ [prev.cudaPackages_11_6.cudnn];

              # Try to force compilation with CUDNN
              # Require compute 8.6 from Ampere: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#improved_fp32
              cmakeFlags =
                oldAttrs.cmakeFlags
                ++ [
                  "-DWITH_CUDNN=ON"
                  "-DOPENCV_DNN_CUDA=ON"
                  "-DCUDA_ARCH_BIN=8.6"
                  "-DOPENCV_ENABLE_NONFREE=ON"
                  "-DCPU_BASELINE=DETECT"
                  "-DCPU_DISPATCH="
                ];

              # We run in a headless environment
              preConfigure =
                oldAttrs.preConfigure
                + ''
                  export ENABLE_HEADLESS=1
                  export CFLAGS="-march=native"
                  export CXXFLAGS="-march=native"
                '';
            });
            # TODO: Tries to download a model immediately upon running, but can't put it in the nix store.
            lpips = with prev.python310Packages;
              buildPythonPackage rec {
                pname = "lpips";
                version = "0.1.4";

                propagatedBuildInputs = [setuptools torchvision tqdm];

                src = prev.python310.pkgs.fetchPypi {
                  inherit pname version;
                  sha256 = "3846331df6c69688aec3d300a5eeef6c529435bc8460bd58201c3d62e56188fa";
                };

                doCheck = false;
              };
          })
        ];
      }; {
        devShells.default = mkShell {
          buildInputs = [
            cudaPackages_11_6.cudatoolkit
            cudaPackages_11_6.cudnn
            (python310.withPackages (ps: with ps; [pytorch torchvision opencv4 pytorch-lightning lpips cupy]))
          ];

          shellHook = ''
            export CUDA_PATH=${cudaPackages.cudatoolkit}
            export LD_LIBRARY_PATH=${linuxPackages_5_15.nvidiaPackages.beta}/lib
            export TORCH_CUDA_ARCH_LIST="8.6"
          '';
        };
      });
}
