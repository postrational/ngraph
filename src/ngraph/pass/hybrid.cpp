//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/pass/hybrid.hpp"
#include "ngraph/op/function_call.hpp"
// #include "hybrid/hybrid_util.hpp"
// #include "ngraph/op/constant.hpp"
// #include "ngraph/op/convert.hpp"
#include "ngraph/op/get_output_element.hpp"
// #include "ngraph/op/slice.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/env_util.hpp"
// #include "ngraph/pass/visualize_tree.hpp"
// #include "nnp_env_vars.hpp"
// #include "nnp_placement.hpp"
// #include "op/function_call.hpp"
// #include "pass/placement.hpp"
// #include "util/layout_util.hpp"
// #include "util/visualize_util.hpp"

using namespace std;
using namespace ngraph;

namespace
{
    template <typename T>
    void insert_in_vector(std::vector<T>& v, T obj)
    {
        if (find(v.begin(), v.end(), obj) == v.end())
        {
            v.push_back(obj);
        }
    }
}

Node* ngraph::pass::Hybrid::take_independent_node_with_placement_priority(
    vector<deque<Node*>>& independent_nodes_by_placement, size_t placement)
{
    Node* selected_node = nullptr;
    NGRAPH_CHECK(placement < independent_nodes_by_placement.size(), "placement out of bounds");
    deque<Node*>& independent_nodes = independent_nodes_by_placement[placement];
    if (independent_nodes.size() != 0)
    {
        selected_node = independent_nodes.front();
        independent_nodes.pop_front();
    }
    else
    {
        for (deque<Node*>& nodes : independent_nodes_by_placement)
        {
            if (nodes.size() > 0)
            {
                selected_node = nodes.front();
                nodes.pop_front();
                break;
            }
        }
    }
    return selected_node;
}

vector<unordered_set<shared_ptr<Node>>>
    ngraph::pass::Hybrid::group_function_nodes_to_clusters(const shared_ptr<Function>& f)
{
    // Topologically sort nodes by picking independent node with the same placement as the
    // previously picked node greedily
    vector<deque<Node*>> independent_nodes_by_placement(2);
    unordered_map<Node*, size_t> node_dependency_count;
    unordered_map<Node*, shared_ptr<Node>> node_map;

    for (shared_ptr<Node> node : f->get_ops())
    {
        size_t dependency_count = node->get_arguments().size();
        node_map[node.get()] = node;
        node_dependency_count[node.get()] = dependency_count;
        if (dependency_count == 0)
        {
            independent_nodes_by_placement[get_placement(node)].push_back(node.get());
        }
    }

    list<shared_ptr<Node>> sorted_nodes;
    // Set previous_placement to 1 so that we start by collecting all of the host
    // fallback nodes first.
    size_t previous_placement = 1;
    while (Node* independent_node = this->take_independent_node_with_placement_priority(
               independent_nodes_by_placement, previous_placement))
    {
        previous_placement = get_placement(independent_node);
        sorted_nodes.push_back(node_map.at(independent_node));

        for (auto user : independent_node->get_users(true))
        {
            Node* user_node = user.get();
            node_dependency_count.at(user_node) -= 1;
            if (node_dependency_count.at(user_node) == 0)
            {
                independent_nodes_by_placement[get_placement(user_node)].push_back(user_node);
            }
        }
    }

    if (sorted_nodes.size() != f->get_ops().size())
    {
        throw ngraph_error("sorted_nodes.size()== " + to_string(sorted_nodes.size()) +
                           " != f->get_ops().size()== " + to_string(f->get_ops().size()) +
                           ". Internal error with topological sort.");
    }

    // Build clusters from the sorted_nodes
    previous_placement = Node::placement_invalid;
    vector<unordered_set<shared_ptr<Node>>> clusters;
    clusters.push_back(unordered_set<shared_ptr<Node>>());

    for (shared_ptr<Node> node : sorted_nodes)
    {
        size_t node_placement = get_placement(node);
        if (node_placement != previous_placement)
        {
            // Since the current node has different placement than the previous node then we
            // need to start a new cluster.
            unordered_set<shared_ptr<Node>> new_cluster;
            clusters.push_back(new_cluster);
        }
        clusters.back().insert(node);
        previous_placement = node_placement;
    }

    // Sanity check for node duplication and full node coverage
    unordered_set<shared_ptr<Node>> cluster_nodes;
    for (auto cluster : clusters)
    {
        for (auto node : cluster)
        {
            if (cluster_nodes.find(node) != cluster_nodes.end())
            {
                throw ngraph_error("Node " + node->get_name() + " is duplicated in clusters");
            }
            cluster_nodes.insert(node);
        }
    }
    unordered_set<shared_ptr<Node>> f_nodes;
    for (auto node : f->get_ordered_ops())
    {
        f_nodes.insert(node);
    }
    if (cluster_nodes != f_nodes)
    {
        throw ngraph_error(
            "Cluster's nodes are not the same as function's nodes. cluster_nodes.size()=" +
            to_string(cluster_nodes.size()) + ", f_nodes.size()=" + to_string(f_nodes.size()));
    }

    return clusters;
}

ngraph::pass::Hybrid::Hybrid(shared_ptr<runtime::Backend> fallback_backend)
    : m_fallback_backend{fallback_backend}
{
}

bool ngraph::pass::Hybrid::run_on_function(std::shared_ptr<ngraph::Function> func)
{
    // Assign GetOutputElement to be on the same device as its parent node
    for (auto node : func->get_ops())
    {
        if (node->description() == "GetOutputElement")
        {
            auto parent = node->get_arguments().at(0);
            node->set_placement(parent->get_placement());
        }
    }

    rewrite_function(func);

    return false;
}

/// \brief Rewrite graph to place fallback nodes within a FunctionCall op
/// \details
///   Assuming the simplest of graphs that has only unary ops as an example. Node A can
///   only run on device while node B can only run on host. P is a Parameter
///
///       P
///       |
///       A
///       |
///       B
///       |
///       B
///       |
///       A
///
///   rewrite_function rewrites the graph creating FunctionCall nodes to replace subgraphs
///   of host fallback nodes, so after rewrite we get this
///
///       P
///       |
///       A
///       |
///  FunctionCall
///       |
///       A
///
///   where FunctionCall contains the subgraph
///
///       P
///       |
///       B
///       |
///       B
///
///   Notice that a Parameter node was appended to the subgraph B-B making it a valid
///   Function that can be passed to the host.
///
///   As the host subgraphs get more complex FunctionCall can have any number of
///   inputs and outputs
///
void ngraph::pass::Hybrid::rewrite_function(const shared_ptr<Function>& f)
{
    // Split functions to clusters of nodes that can be computed together
    vector<unordered_set<shared_ptr<Node>>> clusters = group_function_nodes_to_clusters(f);

    unordered_map<shared_ptr<Node>, unordered_set<shared_ptr<Node>>*> map_node_to_cluster;
    for (auto& cluster : clusters)
    {
        if (cluster.size() > 0)
        {
            shared_ptr<Node> tmp_node = *cluster.begin();
            if (tmp_node == nullptr)
            {
                throw runtime_error("cluster contains nullptr instead of nodes");
            }
            if (is_fallback(tmp_node))
            {
                // This is a non-native cluster so make it a FunctionCall
                vector<Output<Node>> function_call_inputs;
                vector<Output<Node>> function_call_outputs;
                // cluster_inputs contains an ordered list of input Parameters in the cluster.
                // The order of this list matches the order of function_call_inputs
                vector<shared_ptr<op::Parameter>> cluster_inputs;
                vector<Output<Node>> cluster_outputs;
                // input_set contains all of the Output<Node> that are sources of input to this
                // cluster
                vector<Output<Node>> input_set;
                // output_set contains all of the Output<Node> that are sources of output from
                // this cluster
                vector<Output<Node>> output_set;
                map<Output<Node>, shared_ptr<Node>> input_parameter_map;
                for (const shared_ptr<Node>& node : cluster)
                {
                    for (const Input<Node>& input : node->inputs())
                    {
                        Output<Node> source_output = input.get_source_output();
                        Node* source_node = source_output.get_node();
                        if (!is_fallback(source_node) && !source_node->is_constant())
                        {
                            insert_in_vector(input_set, source_output);
                        }
                    }
                    for (const Output<Node>& output : node->outputs())
                    {
                        for (Input<Node> target_input : output.get_target_inputs())
                        {
                            Node* target_node = target_input.get_node();
                            if (!is_fallback(target_node))
                            {
                                insert_in_vector(output_set, output);
                            }
                        }
                    }
                }
                for (const Output<Node>& input : input_set)
                {
                    // Create a collection of input parameters to be used by the new Function
                    // Create a map of original input to new parameter
                    Node* input_node = input.get_node();
                    auto new_parameter = make_shared<ngraph::op::Parameter>(
                        input.get_element_type(), input.get_shape());
                    new_parameter->set_placement(Placement::CPU);
                    new_parameter->set_friendly_name("Mapped from " + input_node->get_name());
                    input_parameter_map.insert({input, new_parameter});

                    cluster_inputs.push_back(new_parameter);
                    function_call_inputs.push_back(input);
                }

                // Rewire the cluster inputs to point to the new Parameters
                for (const Output<Node>& output : input_set)
                {
                    for (const Input<Node>& input : output.get_target_inputs())
                    {
                        shared_ptr<Node> mapped =
                            input_parameter_map.at(output.get_node_shared_ptr());
                        Output<Node> mapped_output = mapped->output(output.get_index());
                        if (is_fallback(input.get_node()) == is_fallback(mapped))
                        {
                            input.replace_source_output(mapped_output);
                        }
                    }
                }
                for (Output<Node> output : output_set)
                {
                    set<Input<Node>> inputs = output.get_target_inputs();
                    cluster_outputs.push_back(output);
                    function_call_outputs.push_back(output);
                }

                // Now make a FunctionCall out of the nodes in cluster, including the new nodes
                // we just added
                auto sub_function = make_shared<Function>(cluster_outputs, cluster_inputs);
                auto fc = make_shared<op::FunctionCall>(
                    function_call_outputs, function_call_inputs, *sub_function, m_fallback_backend);
                fc->set_placement(Placement::CPU);

                // Now connect all of the nodes which get inputs from nodes that now reside inside
                // the FunctionCall we just created
                size_t output_index = 0;
                for (Output<Node> output : output_set)
                {
                    auto input_node = output.get_node();
                    auto index = output.get_index();
                    // auto nnp_layout = dynamic_pointer_cast<runtime::nnp::NNPLayout>(
                    //     input_node->output(index).get_tensor().get_tensor_layout());
                    // NODE_VALIDATION_CHECK(input_node, nnp_layout, "Layout is not NNPLayout");
                    for (Input<Node> input : output.get_target_inputs())
                    {
                        Output<Node> new_output = fc->outputs()[output_index];
                        // runtime::nnp::util::layout_util::set_output_nnp_layout(
                        //     fc, nnp_layout, output_index);
                        auto goe = make_shared<op::GetOutputElement>(fc, output_index);
                        goe->set_placement(Placement::CPU);
                        input.replace_source_output(goe);
                        // Inherit layout from GOE's input
                        // runtime::nnp::util::layout_util::set_output_nnp_layout(goe, nnp_layout, 0);
                    }
                    output_index++;
                }
            }
        }
    }
}

void ngraph::pass::Hybrid::add_hybrid_to_pass_manager(ngraph::pass::Manager& pass_manager)
{
    const std::string backend_name =
        getenv_bool("NGRAPH_GPU_INTERPRETER_FALLBACK") ? "INTERPRETER" : "CPU";
    shared_ptr<runtime::Backend> fallback_backend = runtime::Backend::create(backend_name);
    pass_manager.register_pass<ngraph::pass::Hybrid>(fallback_backend);
}

bool ngraph::pass::Hybrid::is_fallback(const Node* node) const
{
    return node->get_placement() == Placement::CPU;
}

bool ngraph::pass::Hybrid::is_fallback(std::shared_ptr<Node> node) const
{
    return is_fallback(node.get());
}

size_t ngraph::pass::Hybrid::get_placement(const Node* node) const
{
    return is_fallback(node) ? 1 : 0;
}

size_t ngraph::pass::Hybrid::get_placement(std::shared_ptr<Node> node) const
{
    return get_placement(node.get());
}
