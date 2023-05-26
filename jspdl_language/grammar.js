module.exports = grammar({
	name: 'jspdl',

	rules: {
		// TODO: add the actual grammar rules
		program: $ => repeat(choice(
			$._statement,
			$.function_declaration
		)),

		type: $ => choice(
			'int',
			'boolean',
			'string'
		),
		block: $ => seq('{', repeat($._statement), '}'),
		_statement: $ => choice(
			$.let_statement,
			$.if_statement,
			$.do_while_statement,
			$.return_statement,
			$.print_statement,
			$.input_statement,
			$.function_call,
			$.assignment_statement,
			$.post_increment_statement
		),
		let_statement: $ => seq('let', $.type, $.identifier, ';'),
		if_statement: $ => seq('if', '(', field("if_condition", $._expression), ')', $._statement),
		do_while_statement: $ =>
			seq('do', '{', repeat($._statement), '}',
				'while', '(', field("do_while_condition", $._expression), ')', ';'),
		return_statement: $ => seq('return', optional($.return_value), ';'),
		return_value: $ => $._expression,
		print_statement: $ => seq('print', '(', $._expression, ')', ';'),
		input_statement: $ => seq('input', '(', $.identifier, ')', ';'),
		function_call: $ => seq($.identifier, '(', optional($.argument_list), ')', ';'),
		assignment_statement: $ => seq($.identifier, '=', $._expression, ';'),
		post_increment_statement: $ => seq($.identifier, '++', ';'),

		argument_list: $ => seq($._expression, repeat(seq(',', $._expression))),

		function_declaration: $ => seq(
			'function',
			$.identifier,
			optional($.type),
			$.argument_declaration_list,
			$.block
		),

		argument_declaration_list: $ =>
			seq('(',
				optional($.argument_declaration),
				optional(repeat(seq(',', $.argument_declaration))),
				')'),
		argument_declaration: $ =>
			seq(field('type', $.type),
				field('name', $.identifier),),

		parenthesized_expression: $ => seq(
			'(',
			$._expression,
			')'
		),

		_expression: $ => choice(
			$._value,
			$.or_expression,
			$.equality_expression,
			$.addition_expression
		),

		or_expression: $ => prec.left(1, seq($._expression, '||', $._expression)),    // Logical
		equality_expression: $ => prec.left(2, seq($._expression, '==', $._expression)),    // Equality
		addition_expression: $ => prec.left(3, seq($._expression, '+', $._expression)),     // Addition

		_value: $ => choice(
			$.expression_value,
			$.function_call,
			$.parenthesized_expression,
			$.identifier
		),

		expression_value: $ => choice(
			alias(seq($.identifier, ('++')), "post_increment"),
			alias(/-?\d+/, "literal_number"),
			alias(/"[^"]*"/, "literal_string"),
			alias(choice('true', 'false'), "literal_boolean")
		),

		identifier: $ => /[a-zA-Z_]\w*/
	},

	extras: $ => [
		/\s/,
		/\n/,
		/\xa0/, // non-breaking space
		/\/\/.*/, // single-line comment
		/\/\*[^*]*\*+(?:[^/*][^*]*\*+)*\//, // multi-line comment
	],

});