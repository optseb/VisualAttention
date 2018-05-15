<?xml version="1.0" encoding="ISO-8859-1"?><xsl:stylesheet version="1.0"
xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
xmlns:fn="http://www.w3.org/2005/xpath-functions" exclude-result-prefixes="fn">
<xsl:output method="xml" omit-xml-declaration="no" version="1.0" encoding="UTF-8" indent="yes"/>

<!--
     This template adds the Processes and Links necessary to bring
     input data into the Ventral Stream model.

Some DataML help:

This stuff:
        <State c="z" a="data;repeat;" Format="DataML" Version="5" AuthTool="SystemML Toolbox" AuthToolVersion="0">
		<m b="1 1" c="d">0</m>
		<m c="l">1</m>
	</State>

Says that State is a DataMLNode which is of type "z" - the attribute
'c' it used to mean type. 'z' means that the type is struct. The State
node contains some sub-members called "data" and "repeat". Ignore the
rest of the attributes of State (Format, Version Authtool and
AuthToolVersion).

Then go into State to look at "data" and "repeat", in that order. an
<m/> element is a sub-member. Again, attribute c is type. Attribute b
is the dimensions of the thing that <m/> refers to. b="1 1" is a 1x1
array. c="d" means it's a DOUBLE. So, data is 1 by 1 DOUBLE with value
0; repeat is a value rather than an array and is of type BOOL8, which
is what c="l" means, and value 1 a.k.a. true.

Cryptic. Here are all the types (the options for c attributes):

switch(c)
{
case 'd': cache.type = TYPE_DOUBLE; break;
case 'f': cache.type = TYPE_SINGLE; break;

case 'v': cache.type = TYPE_UINT64; break;
case 'u': cache.type = TYPE_UINT32; break;
case 't': cache.type = TYPE_UINT16; break;
case 's': cache.type = TYPE_UINT8; break;

case 'p': cache.type = TYPE_INT64; break;
case 'o': cache.type = TYPE_INT32; break;
case 'n': cache.type = TYPE_INT16; break;
case 'm': cache.type = TYPE_INT8; break;

case 'l': cache.type = TYPE_BOOL8; break;
case 'c': cache.type = TYPE_CHAR16; break;

case 'y': cache.type = TYPE_CELL; break;
case 'z': cache.type = TYPE_STRUCT; break;

See brahms-c++-common.h for more details.


Heres' how it could have looked:

<State type="struct" members="data;repeat;" Format="DataML" Version="5" AuthTool="SystemML Toolbox" AuthToolVersion="0">
  <member arraysize="1 1" type="double">0</member>
  <member type="bool">1</member>
</State>

-->

<xsl:param name="spineml_output_dir" select="'./'"/>

<!-- START TEMPLATE -->
<xsl:template name="external">

<xsl:comment>Generated by external.xsl</xsl:comment>

<!-- GET A LINK TO THE EXPERIMENT FILE FOR LATER USE -->
<xsl:variable name="expt_root" select="/"/>

<!-- GET THE SAMPLE RATE -->
<xsl:variable name="sampleRate" select="(1 div number($expt_root//@dt)) * 1000.0"/>

<!-- GET A LINK TO THE NETWORK FILE FOR LATER USE -->
<!-- <xsl:variable name="main_root" select="/"/> -->

<!-- Zeroes to send as rotations into vsDataMaker -->
<Process>
	<Name>Zeroes</Name>
	<Class>std/2009/source/numeric</Class>
	<Time><SampleRate><xsl:value-of select="$sampleRate"/></SampleRate></Time>
	<State c="z" a="data;repeat;" Format="DataML" Version="5" AuthTool="SystemML Toolbox" AuthToolVersion="0">
		<m b="6 1" c="d">0 0 0 0 0 0</m>
		<m c="l">1</m>
	</State>
</Process>

<!-- This is the world data maker process. It's possible to feed rotations into this process. -->
<Process>
	<Name>vsDataMaker</Name>
	<Class>dev/abrg/vsDataMaker</Class>
	<State c="z" a="output_data_path;neuronsPerPopulation;" Format="DataML" Version="5" AuthTool="SystemML Toolbox" AuthToolVersion="0">
		<m><xsl:value-of select="$spineml_output_dir"/></m>
                <m c="f">22500</m>
	</State>
	<Time><SampleRate><xsl:value-of select="$sampleRate"/></SampleRate></Time>
	<State></State>
</Process>

<!-- Rotations output from Saccsim is input to vsDataMaker component -->
<Link>
	<Src>Zeroes&gt;out</Src>
	<Dst>vsDataMaker&lt;&lt;&lt;rotationsIn</Dst>
	<Lag>0</Lag>
</Link>

<!-- Output from WorldDataMaker is fed into the World population. NB:
     World<in MUST be a receive input and not a reduce input. For some
     reason. -->
<Link>
	<Src>vsDataMaker&gt;corticalSheet</Src>
	<Dst>World&lt;in</Dst>
	<Lag>0</Lag>
</Link>


<!-- END TEMPLATE -->
</xsl:template>

</xsl:stylesheet>
