#!/bin/bash

set -e

# setup local hooks
[ -d .git ] && git config core.hooksPath .githooks

DT_SRC_DIR=$(dirname "$0")
DT_SRC_DIR=$(cd "$DT_SRC_DIR" && pwd -P)


# ---------------------------------------------------------------------------
# Set default values to option vars
# ---------------------------------------------------------------------------

INSTALL_PREFIX_DEFAULT="/opt/ansel"
INSTALL_PREFIX="$INSTALL_PREFIX_DEFAULT"
BUILD_TYPE_DEFAULT="Release"
BUILD_TYPE="$BUILD_TYPE_DEFAULT"
BUILD_DIR_DEFAULT="$DT_SRC_DIR/build"
BUILD_DIR="$BUILD_DIR_DEFAULT"
BUILD_GENERATOR_DEFAULT="Ninja"
BUILD_GENERATOR="$BUILD_GENERATOR_DEFAULT"
BUILD_PACKAGE=0
GIT_UPDATE=0
MAKE_TASKS=-1
ADDRESS_SANITIZER=0
CC_COMPILER="gcc"
CXX_COMPILER="g++"
DO_CLEAN_BUILD=0
DO_CLEAN_INSTALL=0
MANIFEST_FILE="$BUILD_DIR/install_manifest.txt"
FORCE_CLEAN=0
DO_CONFIG=1
DO_BUILD=1
DO_INSTALL=0
SUDO=""

PRINT_HELP=0

FEATURES="COLORD GRAPHICSMAGICK IMAGEMAGICK KWALLET LIBSECRET LUA MAP MAC_INTEGRATION NLS OPENCL OPENEXR OPENMP WEBP"

# prepare a lowercase version with a space before and after
# it's very important for parse_feature, has no impact in for loop expansions
FEATURES_=$(for i in $FEATURES ; do printf " $(printf $i|tr A-Z a-z) "; done)

# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------

parse_feature()
{
	local feature="$1"
	local value="$2"

	if printf "$FEATURES_" | grep -q " $feature " ; then
		eval "FEAT_$(printf $feature|tr a-z A-Z)"=$value
	else
		printf "warning: unknown feature '$feature'\n"
	fi
}

parse_args()
{
	while [ "$#" -ge 1 ] ; do
		option="$1"
		case $option in
		--clean-build)
			DO_CLEAN_BUILD=1
			;;
		--clean-install)
			DO_CLEAN_INSTALL=1
			;;
		--clean-all)
			DO_CLEAN_BUILD=1
			DO_CLEAN_INSTALL=1
			;;
		-f|--force)
			FORCE_CLEAN=1
			;;
		--prefix)
			INSTALL_PREFIX="$2"
			shift
			;;
		--build-type|--buildtype)
			BUILD_TYPE="$2"
			shift
			;;
		--build-dir)
			BUILD_DIR="$2"
			shift
			;;
		--build-generator)
			BUILD_GENERATOR="$2"
			shift
			;;
		--cccompiler)
			CC_COMPILER="$2"
			shift
			;;
		--cxxcompiler)
			CXX_COMPILER="$2"
			shift
			;;
		-j|--jobs)
			MAKE_TASKS=$(printf "%d" "$2" >/dev/null 2>&1 && printf "$2" || printf "$MAKE_TASKS")
			shift
			;;
		--enable-*)
			feature=${option#--enable-}
			parse_feature "$feature" 1
			;;
		--disable-*)
			feature=${option#--disable-}
			parse_feature "$feature" 0
			;;
		--asan)
			ADDRESS_SANITIZER=1
			;;
		--skip-config)
			DO_CONFIG=0
			;;
		--skip-build)
			DO_BUILD=0
			;;
		--install)
			DO_INSTALL=1
			;;
		--sudo)
			SUDO="sudo "
			;;
		--build-package)
			BUILD_PACKAGE=1
			;;
		--update)
			GIT_UPDATE=1
			;;
		-h|--help)
			PRINT_HELP=1
			;;
		*)
			echo "warning: ignoring unknown option $option"
			;;
		esac
		shift
	done
}

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

print_help()
{
	cat <<EOF
$(basename $0) [OPTIONS]

Options:
Installation:
   --prefix         <string>  Install directory prefix
                              (default: $INSTALL_PREFIX_DEFAULT)
   --sudo                     Use sudo when doing the install

Build:
   --build-dir      <string>  Building directory
                              (default: $BUILD_DIR_DEFAULT)
   --build-type     <string>  Build type (Release, Debug, RelWithDebInfo)
                              (default: $BUILD_TYPE_DEFAULT)
   --build-generator <string> Build tool
                              (default: Ninja)
	 --build-package            Build a binary package with only generic optimizations, for portability.
															(default: disabled)

-j --jobs <integer>           Number of tasks
                              (default: number of CPUs)

   --asan                     Enable address sanitizer options
                              (default: disabled)

	 --cccompiler               C Compiler (default: gcc)
	                            (alternative: clang)

	 --cxxcompiler              C++ Compiler (default: g++)
	                            (alternative: clang++)

Actual actions:
   --skip-build               Configure but exit before building the binaries
                              (default: disabled)
   --install                  After building the binaries, install them
                              (default: disabled)

Cleanup actions:
   --clean-build              Clean build directory
   --clean-install            Clean install directory
   --clean-all                Clean both build and install directories
-f --force                    Force clean-build to perform removal
                              ignoring any errors

Update actions:
   --update                   Run 'git pull' to update the source code and submodules
	                            from the project master branch.
	                            Git needs to be installed on the computer.
															(default: disabled)


Features:
By default cmake will enable the features it autodetects on the build machine.
Specifying the option on the command line forces the feature on or off.
All these options have a --disable-* equivalent.
$(for i in $FEATURES_ ; do printf "    --enable-$i\n"; done)

Extra:
-h --help                Print help message
EOF

}

# ---------------------------------------------------------------------------
# utility functions
# ---------------------------------------------------------------------------

log()
{
	local prefix
	case $1 in
		trace) prefix="[\x1b[32mTRACE\x1b[0m] " ;;
		debug|dbg) prefix="[\x1b[35mDEBUG\x1b[0m] " ;;
		info) prefix="[\x1b[36mINFO\x1b[0m] " ;;
		warning|warn) prefix="[\x1b[33mWARNING\x1b[0m] " ;;
		error|err) prefix="[\x1b[31mERROR\x1b[0m] " ;;
		critical) prefix="[\x1b[31;01mCRITICAL\x1b[0m] " ;;
	esac

	echo -e "$2" |sed -e "s/^/$prefix/"
}

num_cpu()
{
	local ncpu
	local platform=$(uname -s)

	case "$platform" in
	SunOS)
		ncpu=$(/usr/sbin/psrinfo |wc -l)
		;;
	Linux|MINGW64*)
		if [ -r /proc/cpuinfo ]; then
			ncpu=$(grep -c "^processor" /proc/cpuinfo)
		elif [ -x /sbin/sysctl ]; then
			ncpu=$(/sbin/sysctl -n hw.ncpu 2>/dev/null)
			if [ $? -neq 0 ]; then
				ncpu=-1
			fi
		fi
		;;
	Darwin)
		ncpu=$(/usr/sbin/sysctl -n machdep.cpu.core_count 2>/dev/null)
		;;
	*)
		printf "warning: unable to determine number of CPUs on $platform\n"
		ncpu=-1
		;;
	esac

	if [ $ncpu -lt 1 ] ; then
		ncpu=1
	fi
	printf "$ncpu"
}

make_name()
{
	local make="make"
	local platform=$(uname -s)

	case "$platform" in
	SunOS)
		PATH="/usr/gnu/bin:$PATH"
		export PATH
		make="gmake"
		;;
	esac
	printf "$make"
}

features_set_to_autodetect()
{
	for i in $FEATURES; do
		eval FEAT_$i=-1
	done
}

cmake_boolean_option()
{
	name=$1
	value=$2
	case $value in
	-1)
		# Do nothing
		;;
	0)
		CMAKE_MORE_OPTIONS="$CMAKE_MORE_OPTIONS -D${name}=Off"
		;;
	1)
		CMAKE_MORE_OPTIONS="$CMAKE_MORE_OPTIONS -D${name}=On"
		;;
	esac
}

clean_build()
{
	local force=$1
	local path_to_clean=$2
	local option="-I"

	[ $force -eq 1 ] && option="-f"

	${SUDO}rm -r "$option" "$path_to_clean" || log err "Failed to remove [$path_to_clean]"
}

clean_install()
{
	local force=$1
	local file=$2
	local option="-I"

	[ $force -eq 1 ] && option="-f"
	${SUDO}rm "$option" "$file" && log info "Removed: $file"
}

# ---------------------------------------------------------------------------
# Let's process the user's wishes
# ---------------------------------------------------------------------------

MAKE_TASKS=$(num_cpu)
MAKE=$(make_name)

features_set_to_autodetect
parse_args "$@"

if [ $PRINT_HELP -ne 0 ] ; then
	print_help
	exit 1
fi

CMAKE_MORE_OPTIONS=""
for i in $FEATURES; do
	eval cmake_boolean_option USE_$i \$FEAT_$i
done

# Some people might need this, but ignore if unset in environment
CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-}
CMAKE_MORE_OPTIONS="${CMAKE_MORE_OPTIONS} ${CMAKE_PREFIX_PATH}"

# ---------------------------------------------------------------------------
# Update source code and submodules
# ---------------------------------------------------------------------------

if [ $GIT_UPDATE -eq 1 ] ;
then
	git pull --recurse-submodules
fi


# ---------------------------------------------------------------------------
# Determine CPU architecture
# ---------------------------------------------------------------------------

CPU_ARCHITECTURE=""
if [[ `uname -a` =~ ^Darwin.* ]] && [[ `uname -a` =~ .*arm64$ ]]
then
    CPU_ARCHITECTURE="ARM64"
    CMAKE_MORE_OPTIONS="${CMAKE_MORE_OPTIONS} -DBUILD_SSE2_CODEPATHS=OFF"
else
	CPU_ARCHITECTURE="Intel"
fi

# ---------------------------------------------------------------------------
# Generic package or customized build ?
# ---------------------------------------------------------------------------

if [ $BUILD_PACKAGE -eq 1 ] ;
then
	CMAKE_MORE_OPTIONS="${CMAKE_MORE_OPTIONS} -DBINARY_PACKAGE_BUILD=ON"
else
	CMAKE_MORE_OPTIONS="${CMAKE_MORE_OPTIONS} -DBINARY_PACKAGE_BUILD=OFF"
fi

# ---------------------------------------------------------------------------
# Let's go
# ---------------------------------------------------------------------------

cat <<EOF
Ansel build script

Building directory:  $BUILD_DIR
Installation prefix: $INSTALL_PREFIX
Build type:          $BUILD_TYPE
Build generator:     $BUILD_GENERATOR
Build tasks:         $MAKE_TASKS
CPU Architecture:    $CPU_ARCHITECTURE
Compiler:            $CC_COMPILER $CXX_COMPILER

EOF

# ---------------------------------------------------------------------------
# Let's clean some things
# ---------------------------------------------------------------------------
if [ $DO_CLEAN_INSTALL -gt 0 ] ; then
	log info "Cleaning installation directory from $MANIFEST_FILE"
	if [ -f $MANIFEST_FILE ]; then
		for f in $(cat $MANIFEST_FILE); do
			if [ -f "$f" ]; then
				clean_install $FORCE_CLEAN "$f"
			else
				log warn "File not found: can't remove $f"
			fi
		done
	else
		log err "File not found: $MANIFEST_FILE"
	fi
fi

if [ $DO_CLEAN_BUILD -gt 0 ] ; then
	echo
	log warn "Cleaning directory ["$BUILD_DIR"]: it will erase all the files in this path"
	clean_build $FORCE_CLEAN "$BUILD_DIR"
fi


# ---------------------------------------------------------------------------
# CMake
# ---------------------------------------------------------------------------

mkdir -p "$BUILD_DIR"

if [ $ADDRESS_SANITIZER -ne 0 ] ; then
	ASAN_FLAGS="CFLAGS=\"-fsanitize=address -fno-omit-frame-pointer\""
	ASAN_FLAGS="$ASAN_FLAGS CXXFLAGS=\"-fsanitize=address -fno-omit-frame-pointer\""
	ASAN_FLAGS="$ASAN_FLAGS LDFLAGS=\"-fsanitize=address\" "
fi


cmd_config="CXX=${CXX_COMPILER} CC=${CC_COMPILER} ${ASAN_FLAGS}cmake -G \"$BUILD_GENERATOR\" -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ${CMAKE_MORE_OPTIONS} \"$DT_SRC_DIR\""
cmd_build="cmake --build "$BUILD_DIR" -- -j$MAKE_TASKS"
cmd_install="${SUDO}cmake --build \"$BUILD_DIR\" --target install -- -j$MAKE_TASKS"

cat <<EOF

Complete build options:
$cmd_config

EOF

OLDPWD="$(pwd)"

if [ $DO_CONFIG -eq 0 ] ; then
	cat <<EOF
The script would have configured, built, and installed with these commands:
\$ $(printf "$cmd_config")
\$ $(printf "$cmd_build")
\$ $(printf "$cmd_install")
EOF
	exit 0
fi

# configure the build
cd "$BUILD_DIR"
eval "$cmd_config"
cd "$OLDPWD"


if [ $DO_BUILD -eq 0 ] ; then
	cat <<EOF
The ansel configuration is finished.
To build and install ansel you need to type:
\$ $(printf "$cmd_build")
\$ $(printf "$cmd_install")
EOF
	exit 0
fi

# build the binaries
eval "$cmd_build"

if [ $DO_INSTALL -eq 0 ] ; then
	cat <<EOF
ansel finished building.
To actually install ansel you need to type:
\$ $(printf "$cmd_install")
EOF
	exit 0
fi

# install the binaries
eval "$cmd_install"

# install the desktop launcher and system-wide command
if [ $DO_INSTALL ] ; then
	if [ -f "$INSTALL_PREFIX/bin/ansel" ]; then
		[ ! -d "/usr/local/bin/" ] && $SUDO mkdir -p /usr/local/bin/
		[ -f "/usr/local/bin/ansel" ] && $SUDO rm /usr/local/bin/ansel

		$SUDO ln -s "$INSTALL_PREFIX"/bin/ansel /usr/local/bin/ansel
	fi

	if [ -f "$INSTALL_PREFIX/share/applications/photos.ansel.app.desktop" ]; then
		[ ! -d "/usr/share/applications/" ] && $SUDO mkdir -p /usr/share/applications/
		[ -f "/usr/share/applications/ansel.desktop" ] && $SUDO rm /usr/share/applications/ansel.desktop

		$SUDO ln -s "$INSTALL_PREFIX"/share/applications/photos.ansel.app.desktop /usr/share/applications/ansel.desktop
	fi
fi
# update Lensfun
$SUDO lensfun-update-data
