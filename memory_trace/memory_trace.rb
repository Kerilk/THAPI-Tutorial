require 'optparse'
require 'babeltrace2'
require 'find'
require 'pp'

ctf_fs = BT2::BTPlugin.find("ctf").get_source_component_class_by_name("fs")
utils_muxer = BT2::BTPlugin.find("utils").get_filter_component_class_by_name("muxer")

trace_locations = Find.find(*ARGV).reject { |path|
    FileTest.directory?(path)
  }.select { |path|
    File.basename(path) == "metadata"
  }.collect { |path|
    File.dirname(path)
  }.select { |path|
    qe = BT2::BTQueryExecutor.new( component_class: ctf_fs, object_name: "babeltrace.support-info", params: { "input" => path,  "type" => "directory" } )
    qe.query.value["weight"] > 0.5
  }



$stacks = Hash.new { |h, k| h[k] = [] }
consume = lambda { |iterator, _|
  mess = iterator.next_messages
  mess.each { |m|
    if m.type == :BT_MESSAGE_TYPE_EVENT
      e = m.event
      id = [e.stream.trace.get_environment_entry_value_by_name("hostname").value,
            e.get_common_context_field.value]
      if e.name.match(/_entry\z/)
        $stacks[id].push [Time.at(0, m.get_default_clock_snapshot.ns_from_origin, :nsec),
                         e.name[0..-7], e.payload_field.value]
      elsif e.name.match(/_exit\z/)
        time, name, payload = $stacks[id].pop
        cb = $callbacks[name]
        cb.call(id, time, Time.at(0, m.get_default_clock_snapshot.ns_from_origin, :nsec), payload, e.payload_field.value) if cb
      else
        cb = $callbacks[e.name]
        cb.call(id, Time.at(0, m.get_default_clock_snapshot.ns_from_origin, :nsec), e.payload_field.value) if cb
      end
    end
  }
}

graph = BT2::BTGraph.new
comps = trace_locations.each_with_index.collect { |trace_location, i| graph.add_component(ctf_fs, "trace_#{i}", params: {"inputs" => [ trace_location ] }) }

comp2 = graph.add_component(utils_muxer, "mux")
comp3 = graph.add_simple_sink("memory_trace", consume)
i = 0
comps.each { |comp|
  ops = comp.output_ports
  ops.each { |op|
    ip = comp2.input_port(i)
    i += 1
    graph.connect_ports(op, ip)
  }
}

op = comp2.output_port(0)
ip = comp3.input_port(0)
graph.connect_ports(op, ip)

# Callbacks should be of the form:
#  - for regular API events:
#    `lambda { |id, start, stop, payload_in, payload_out| ... }`
#  - for THAPI events:
#    `lambda { |id, timestamp, payload| ... }`
# And hashed by their name (without _entry and _exit suffixes)
# example:
#  `$callbacks["lttng_ust_opencl:clGetContextInfo"] = lambda { |*args| p args }`
$callbacks = {}


graph.run
