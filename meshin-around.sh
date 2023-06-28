#!/bin/bash	

_meshin_around_main() {
	local mesh="${1}" ; shift
	local mash="${1-fun.obj}" ; shift

	if [ "" = "${mesh}" ] ; then
		echo "usage: meshin-around.sh <download.obj>"
		return 1
	fi

	local name=$( basename ${mash} | sed 's,\.obj$,,' )
	local mtl="${name}.mtl"
	local png="${name}.png"

	if [ -f ${mash} ] ; then
		echo "${mash} already exists"
	else
		echo "creating ${mash} for ${mesh}"
		sed "s,^f.*,,;s,#f,f,;s,.*mtllib.*,mtllib ${mtl}," ${mesh} > ${mash} || exit ${?}
	fi

	if [ -f "${mtl}" ] ; then 
		echo "${mtl} already exists"
	else
		echo "creating ${mtl} for ${mash}"
		echo -e "newmtl MyMaterial\nmap_Kd ${png}" > ${mtl} || exit ${?}
	fi

	if [ -f "${png}" ] ; then
		echo "${png} looks good"
	else
		echo "be sure your texture is in pwd and named ${png} or edit ${mtl}"
	fi
}

_meshin_around_main ${*}
